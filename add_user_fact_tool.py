import logging
from typing import Optional, Type
from uuid import UUID, uuid4

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db_vector.cache_backed_embedding import get_cache_backed_embedding
from app.models import UserFacts

logger = logging.getLogger(__name__)


class AddUserFactInput(BaseModel):
    fact: str = Field(
        description="Some fact about the user. Like work address or normal daily start and end time for work.",
    )


class AddUserFactTool(BaseTool):
    name = settings.tool_names.add_user_fact_tool
    description = f"Adds fact about the user to DB"

    async_session: AsyncSession
    user_id: UUID

    args_schema: Type[BaseModel] = AddUserFactInput

    def _run(
        self,
        fact: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        raise NotImplementedError()

    async def _arun(
        self,
        fact: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ):
        try:
            cache_backed_embedding = await get_cache_backed_embedding()
            embedded_fact = await cache_backed_embedding.aembed_query(fact)

            user_fact = UserFacts(
                id=uuid4(), user_id=self.user_id, fact=fact, embedding=embedded_fact
            )

            self.async_session.add(user_fact)
            await self.async_session.commit()

            return "User fact added successfully"
        except Exception as e:
            logger.error(f"Failed adding fact for user {self.creator_id}: {e}")
            return "Adding user fact failed"
