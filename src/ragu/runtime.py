from ragu.auth import AuthService
from ragu.confluence import ConfluenceSource
from ragu.database import Database
from ragu.ingestion import IngestionService
from ragu.local_consumption import LocalConsumptionSource
from ragu.management import ManagementService
from ragu.multi_source import MultiSourceStrategy
from ragu.oauth import OAuthService
from ragu.projects import ProjectService
from ragu.reranking import Reranker
from ragu.search import SearchService
from ragu.strategies import ConfluenceRetrievalStrategy, LocalConsumptionStrategy, StrategyRegistry
from ragu.vectors import Vectors


class Runtime:
    def __init__(
        self, settings, *, db=None, source=None, vectors=None, mock_confluence=None, reranker=None
    ):
        self.settings = settings
        self.db = db or Database(settings.database_url())
        self.auth = AuthService(self.db, settings)
        self.source = source or ConfluenceSource(settings)
        self.local_consumption = LocalConsumptionSource(settings)
        self.vectors = vectors or Vectors(settings)
        self.multi_source = MultiSourceStrategy(
            self.db, self.auth, settings, self.vectors, confluence=mock_confluence
        )
        self.strategies = StrategyRegistry(
            LocalConsumptionStrategy(
                self.db, self.auth, settings, self.local_consumption, self.vectors
            ),
            ConfluenceRetrievalStrategy(self.db, self.auth, settings, self.source, self.vectors),
            self.multi_source,
        )
        self.management = ManagementService(
            self.db, self.auth, settings, self.vectors, self.strategies
        )
        self.projects = ProjectService(
            self.db, self.auth, settings, self.strategies, self.multi_source
        )
        self.ingestion = IngestionService(
            self.db,
            self.auth,
            settings,
            self.vectors,
            self.strategies,
            self.multi_source,
            self.projects,
        )
        self.reranker = reranker or Reranker(settings)
        self.search = SearchService(self.strategies, self.reranker, self.projects)
        self.oauth = OAuthService(self.db, self.auth, settings)

    def close(self):
        self.source.client.close()
        self.vectors.close()
        self.db.engine.dispose()
