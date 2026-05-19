import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssl_fix  # noqa

'''
Pydantic Basics — Runtime Data Validation

Pydantic enforces types and constraints at runtime, not just statically.
When you pass data into a Pydantic model, it:
  1. Coerces values where possible (e.g., "25" -> 25 for an int field)
  2. Validates constraints (e.g., gt=0 means value must be > 0)
  3. Raises a clear ValidationError if data is invalid

Field() gives fine-grained control:
  - default      -> value used when the field is absent
  - gt / lt      -> numeric bounds (greater-than / less-than)
  - description  -> documents the field (also used by LangChain output parsers)
  - min_length   -> minimum string length

.model_dump_json() serialises the validated model to a JSON string —
handy for logging, storing, or sending over an API.
'''

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class CourseEnrollment(BaseModel):
    student_name: str = "Anonymous"
    student_id:   Optional[str] = None
    email:        EmailStr
    gpa:          float = Field(gt=0.0, lt=4.1, default=2.0)
    course:       str   = Field(min_length=3, description="Name of the enrolled course")

# Pydantic coerces "3.8" (string) -> 3.8 (float) automatically
sample_data = {
    "student_name": "Srishant",
    "student_id":   "CS2024001",
    "email":        "srishant@example.com",
    "gpa":          "3.8",   # passed as string — Pydantic coerces it
    "course":       "Deep Learning with PyTorch"
}

enrollment = CourseEnrollment(**sample_data)

print("Validated model:")
print(enrollment)
print()
print("As JSON:")
print(enrollment.model_dump_json(indent=2))
