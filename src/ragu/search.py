from ragu.auth import Principal
from ragu.errors import AppError
from ragu.schemas import DEFAULT_SOURCE, RerankMethod, SearchQuery
from ragu.strategies import StrategyRegistry


class SearchService:
    def __init__(self, strategies: StrategyRegistry, reranker=None, projects=None):
        self.strategies = strategies
        self.reranker, self.projects = reranker, projects

    def _strategy(self, data, principal):
        if data.corpus_id is None:
            return self.strategies.get(data.source)
        if self.projects is None:
            raise AppError(503, "Project retrieval is unavailable.")
        strategy = self.projects.strategy(data.corpus_id, principal)
        if "source" in data.model_fields_set and data.source != strategy.source:
            raise AppError(422, "source does not match the selected corpus.")
        return strategy

    def search(self, data: SearchQuery, principal: Principal) -> dict:
        if not data.query.strip():
            raise AppError(422, "A nonempty query is required.")
        strategy = self._strategy(data, principal)
        if self.reranker is None:
            return strategy.search(data, principal)
        count = data.candidate_limit or (
            max(40, data.limit) if data.rerank_method != RerankMethod.NONE else data.limit
        )
        query = data if count == data.limit else data.model_copy(update={"limit": count})
        batch = strategy.search(query, principal)
        candidates = batch["results"]
        ranked = self.reranker.rank(data.query, candidates, data.rerank_method)
        if data.rerank_method != RerankMethod.NONE and ranked:
            ranked = self._confirm(strategy, query, principal, batch, ranked)
        batch["results"] = ranked[: data.limit]
        batch["candidate_count"] = len(candidates)
        batch["partial"] = len(batch["results"]) < data.limit
        batch["candidate_method"] = data.candidate_method
        batch["rerank_method"] = data.rerank_method
        batch["rerank_model"] = (
            self.reranker.settings.reranking_model
            if data.rerank_method != RerankMethod.NONE
            else None
        )
        strategy.authorize(principal)
        return batch

    def candidates(self, data, principal):
        count = data.candidate_limit or data.limit
        query = data.model_copy(update={"limit": count, "candidate_limit": count})
        return self.search(query, principal)

    def rerank(self, data, principal):
        if not data.query.strip():
            raise AppError(422, "A nonempty query is required.")
        if self.reranker is None:
            raise AppError(503, "Reranking is unavailable.")
        strategy = self._strategy(data, principal)
        query = SearchQuery(query=data.query).model_copy(update={"limit": len(data.candidate_ids)})
        batch = strategy.search(
            query,
            principal,
            candidate_ids=[str(identifier) for identifier in data.candidate_ids],
            generation_id=data.generation_id,
        )
        if len(batch["results"]) != len(data.candidate_ids):
            raise AppError(409, "One or more candidates changed; retrieve candidates again.")
        for item in batch["results"]:
            item["score"], item["score_type"] = None, "not_computed"
        ranked = self.reranker.rank(data.query, batch["results"], data.rerank_method)
        ranked = self._confirm(strategy, query, principal, batch, ranked)
        batch["results"] = ranked[: data.limit]
        batch["candidate_count"] = len(ranked)
        batch["rerank_method"] = data.rerank_method
        batch["rerank_model"] = (
            self.reranker.settings.reranking_model
            if data.rerank_method != RerankMethod.NONE
            else None
        )
        batch["partial"] = False
        strategy.authorize(principal)
        return batch

    @staticmethod
    def _confirm(strategy, query, principal, batch, ranked):
        # Revalidate after inference so revocation, source changes and publication cannot leak stale text.
        refreshed = strategy.search(
            query,
            principal,
            candidate_ids=[item["candidate_id"] for item in ranked],
            generation_id=batch["generation_id"],
        )
        by_id = {item["candidate_id"]: item for item in refreshed["results"]}
        if len(by_id) != len(ranked):
            raise AppError(409, "Candidates changed during reranking; retrieve candidates again.")
        return [
            {
                **by_id[item["candidate_id"]],
                "score": item["score"],
                "score_type": item["score_type"],
                **({"rerank_score": item["rerank_score"]} if "rerank_score" in item else {}),
            }
            for item in ranked
        ]

    def capabilities(self, principal, source=DEFAULT_SOURCE, corpus_id=None):
        strategy = (
            self.projects.strategy(corpus_id, principal)
            if corpus_id is not None
            else self.strategies.get(source)
        )
        collection = strategy._context(principal)[0]
        info = strategy.vectors.info(collection)
        result = {
            "source": strategy.source,
            "corpus_id": corpus_id,
            "generation_id": strategy._generation(collection),
            "candidate_methods": strategy.vectors.methods(collection),
            "index_profile": (info.config.metadata or {}).get("ragu_profile"),
            "rerank_methods": list(RerankMethod),
            "reranker_model": self.reranker.settings.reranking_model,
            "max_candidates": 100,
            "max_results": 20,
        }
        strategy.authorize(principal)
        return result

    def document(
        self, page_id: str, principal: Principal, source: str = DEFAULT_SOURCE, corpus_id=None
    ) -> dict:
        strategy = (
            self.projects.strategy(corpus_id, principal)
            if corpus_id is not None
            else self.strategies.get(source)
        )
        result = strategy.document(page_id, principal)
        if self.reranker is not None:
            strategy.authorize(principal)
        return result
