from pydantic import BaseModel
class Result(BaseModel):
    content: str
    reasoning: str

 