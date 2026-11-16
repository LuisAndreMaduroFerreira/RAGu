import logging

from sqlalchemy import delete, or_, select

from ragu.database import Corpus, CorpusGrant, Database, Project, ProjectMember, User
from ragu.errors import AppError

logger = logging.getLogger("ragu.audit")


class ProjectService:
    def __init__(self, db, auth, settings, strategies, multi_source):
        self.db, self.auth, self.settings = db, auth, settings
        self.strategies, self.multi_source = strategies, multi_source

    def project_access(self, session, project_id, user_id, *, manage=False):
        project = session.get(Project, project_id, with_for_update=True)
        if project is None:
            raise AppError(404, "Project not found or not accessible.")
        if project.owner_id != user_id:
            member = session.get(ProjectMember, (project_id, user_id))
            if member is None or (manage and member.role != "maintainer"):
                raise AppError(404, "Project not found or not accessible.")
        return project

    def corpus_access(self, session, corpus_id, user_id, *, write=False):
        corpus = session.get(Corpus, corpus_id)
        if corpus is None:
            raise AppError(404, "Corpus not found or not accessible.")
        project = self.project_access(session, corpus.project_id, user_id)
        if project.owner_id != user_id:
            grant = session.get(CorpusGrant, (corpus_id, user_id))
            if grant is None or (write and grant.permission != "ingest"):
                raise AppError(404, "Corpus not found or not accessible.")
        return corpus

    @staticmethod
    def _project_view(project):
        return {"id": project.id, "name": project.name, "owner_id": project.owner_id}

    @staticmethod
    def _corpus_view(corpus):
        return {
            "id": corpus.id,
            "project_id": corpus.project_id,
            "name": corpus.name,
            "source": corpus.source,
            "index_mode": corpus.index_mode,
            "published_at": corpus.published_at,
        }

    def create(self, data, principal):
        with self.db.session() as session:
            self.auth.current(session, principal, admin=True)
            project = Project(name=data.name.strip(), owner_id=principal.id)
            session.add(project)
            session.flush()
            logger.info("project_created project_id=%s user_id=%s", project.id, principal.id)
            return self._project_view(project)

    def list(self, principal):
        with self.db.session() as session:
            self.auth.current(session, principal)
            membership = select(ProjectMember.project_id).where(
                ProjectMember.user_id == principal.id
            )
            return [
                self._project_view(item)
                for item in session.scalars(
                    select(Project)
                    .where(or_(Project.owner_id == principal.id, Project.id.in_(membership)))
                    .order_by(Project.created_at)
                    .limit(100)
                )
            ]

    def member(self, project_id, user_id, data, principal):
        with self.db.session() as session:
            Database.lock(session, 1002)
            self.auth.current(session, principal)
            project = self.project_access(session, project_id, principal.id, manage=True)
            if user_id == project.owner_id:
                raise AppError(409, "Project ownership cannot be changed through membership.")
            target = session.get(User, user_id)
            if target is None or (data is not None and not target.is_active):
                raise AppError(404, "Active user not found.")
            membership = session.get(ProjectMember, (project_id, user_id))
            if data is None:
                if membership is not None:
                    session.delete(membership)
                session.execute(
                    delete(CorpusGrant).where(
                        CorpusGrant.user_id == user_id,
                        CorpusGrant.corpus_id.in_(
                            select(Corpus.id).where(Corpus.project_id == project_id)
                        ),
                    )
                )
            elif membership is None:
                session.add(ProjectMember(project_id=project_id, user_id=user_id, role=data.role))
            else:
                membership.role = data.role
            logger.info(
                "project_membership_changed project_id=%s actor_id=%s user_id=%s role=%s",
                project_id,
                principal.id,
                user_id,
                data.role if data is not None else "removed",
            )
            return {"message": "Project membership updated."}

    def create_corpus(self, project_id, data, principal):
        with self.db.session() as session:
            Database.lock(session, 1002)
            self.auth.current(session, principal, admin=True)
            self.project_access(session, project_id, principal.id, manage=True)
            if data.source == "local_consumption":
                root = data.folder or self.settings.local_consumption_root
                if root is None:
                    raise AppError(503, "Configure an approved local consumption root first.")
                self.multi_source._folder(root)
            corpus = Corpus(
                project_id=project_id,
                name=data.name.strip(),
                source=data.source,
                owner_id=principal.id,
                configuration={"folder": str(root)} if data.source == "local_consumption" else {},
            )
            session.add(corpus)
            session.flush()
            if self.project_access(session, project_id, principal.id).owner_id != principal.id:
                session.add(
                    CorpusGrant(corpus_id=corpus.id, user_id=principal.id, permission="ingest")
                )
            logger.info("corpus_created corpus_id=%s user_id=%s", corpus.id, principal.id)
            return self._corpus_view(corpus)

    def corpora(self, project_id, principal):
        with self.db.session() as session:
            self.auth.current(session, principal)
            project = self.project_access(session, project_id, principal.id)
            query = select(Corpus).where(Corpus.project_id == project_id)
            if project.owner_id != principal.id:
                query = query.where(
                    Corpus.id.in_(
                        select(CorpusGrant.corpus_id).where(CorpusGrant.user_id == principal.id)
                    )
                )
            return [self._corpus_view(item) for item in session.scalars(query.limit(100))]

    def grant(self, corpus_id, user_id, data, principal):
        with self.db.session() as session:
            Database.lock(session, 1002)
            self.auth.current(session, principal)
            corpus = session.get(Corpus, corpus_id)
            if corpus is None:
                raise AppError(404, "Corpus not found or not accessible.")
            project = self.project_access(session, corpus.project_id, principal.id, manage=True)
            if user_id == project.owner_id:
                raise AppError(409, "The project owner already has corpus access.")
            self.project_access(session, corpus.project_id, user_id)
            grant = session.get(CorpusGrant, (corpus_id, user_id))
            if data is None:
                if grant is not None:
                    session.delete(grant)
            elif grant is None:
                session.add(
                    CorpusGrant(corpus_id=corpus_id, user_id=user_id, permission=data.permission)
                )
            else:
                grant.permission = data.permission
            logger.info(
                "corpus_grant_changed corpus_id=%s actor_id=%s user_id=%s permission=%s",
                corpus_id,
                principal.id,
                user_id,
                data.permission if data is not None else "removed",
            )
            return {"message": "Corpus permission updated."}

    def strategy(self, corpus_id, principal, *, write=False):
        with self.db.session() as session:
            self.auth.current(session, principal)
            corpus = self.corpus_access(session, corpus_id, principal.id, write=write)
            return self.for_corpus(corpus)

    def for_corpus(self, corpus):
        from pathlib import Path

        from ragu.multi_source import MultiSourceStrategy
        from ragu.strategies import ConfluenceRetrievalStrategy, LocalConsumptionStrategy

        if corpus.source == "local_consumption":
            files = self.multi_source._folder(Path(corpus.configuration["folder"]))
            strategy = LocalConsumptionStrategy(
                self.db, self.auth, self.settings, files, self.multi_source.vectors
            )
        elif corpus.source == "confluence":
            source = self.strategies.get("confluence").source_client
            strategy = ConfluenceRetrievalStrategy(
                self.db, self.auth, self.settings, source, self.multi_source.vectors
            )
        else:
            strategy = MultiSourceStrategy(
                self.db,
                self.auth,
                self.settings,
                self.multi_source.vectors,
                confluence=self.multi_source.confluence,
            )
            strategy.cache_factory = self.multi_source.cache_factory
        strategy.configure_scope(corpus.id, self)
        return strategy
