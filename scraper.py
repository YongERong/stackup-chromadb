from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import dateparser
from scrapegraphai.graphs import SmartScraperGraph
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()
load_dotenv()


class Event(BaseModel):
    name: str = Field("Name of event")
    link: str
    start_date: Union[datetime, None] = Field(
        "Start date of event, put 'null' if not found"
    )
    end_date: Union[datetime, None] = Field(
        "End date of event, put 'null' if not found"
    )
    location: str = Field(
        "Location where event is held, put 'Online' if event is held virtually"
    )
    registration_deadline: Union[datetime, None] = Field(
        "Registration deadline of event, put 'null' if not found"
    )
    description: str = Field(
        "Brief 100 word description of the event, highlight the key themes and skills required"
    )

    @field_validator("start_date", "end_date", "registration_deadline", mode="before")
    @classmethod
    def check_date(cls, v: str) -> datetime:
        if not v:
            return None

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        dateparse_prompt = PromptTemplate.from_template(
            "Extract date and time from the date_text, including timezone if available. Only include the date and time and nothing else.\n date_text: {date_text}"
        )
        dateparse_chain = dateparse_prompt | llm
        date_str = dateparse_chain.invoke({"date_text": v}).content.strip()
        date = dateparser.parse(date_str)

        if date:
            return date
        else:
            raise ValueError(f"Error parsing date string: {date_str}")


# Define the configuration for the scraping pipeline
graph_config = {
    "llm": {
        "api_key": "AIzaSyCQmxiZ74532X72Br0qqU9zeyQT348ya8U",
        "model": "google_genai/gemini-pro",
    },
    "verbose": True,
    "headless": True,
}


def scrape(url):
    smart_scraper_graph = SmartScraperGraph(
        prompt="Obtain information about the event.",
        source=url,
        config=graph_config,
        schema=Event,
    )
    result = Event.model_validate(smart_scraper_graph.run())
    return result
