"""The JSON shapes the model must reply with (checked by pydantic)."""
import re

from pydantic import BaseModel, field_validator


class Part(BaseModel):
    label: str = ""                # "(a)", "(b)(i)", ... (set by Python from the blueprint)
    text: str
    mark_scheme: list[str]         # marking points or working steps; Python adds the B1 / C1 / A1 codes
    unit: str = ""                 # unit of the numerical answer, e.g. "m s^-1"
    calculation: str = ""          # ONE Python expression that gives the numerical answer
    marks: int = 0                 # set by Python from the blueprint
    answer: float | None = None    # set by Python from the calculation

    @field_validator("mark_scheme", mode="before")
    @classmethod
    def lines_from_text(cls, v):   # one long string -> one line per point
        return [line.strip() for line in re.split(r"\n|;", v) if line.strip()] if isinstance(v, str) else v


class NewQuestion(BaseModel):
    stem: str
    figure: str = ""               # words describing a diagram, if one is needed
    graph_x_label: str = ""        # a graph, if one is needed: axis labels, x range, and y as a formula of x
    graph_y_label: str = ""
    graph_x_from: float = 0
    graph_x_to: float = 0
    graph_y_formula: str = ""
    parts: list[Part]


class Review(BaseModel):
    problems: list[str] = []
