import pandas as pd

from src.data.validation.validator import InputValidator
from src.data.validation.rules.generic import RequiredFieldRule, NonEmptyRowRule
from src.data.validation.rules.domain import CreatedBeforeScheduleRule


input_df = pd.read_csv('./data/input.csv')

rules = [
    NonEmptyRowRule(),
    
    RequiredFieldRule("service_id"),
    RequiredFieldRule("labor_id"),
    RequiredFieldRule("created_at"),
    RequiredFieldRule("schedule_date"),
    RequiredFieldRule("start_address_point"),
    RequiredFieldRule("labor_name"),
    RequiredFieldRule("end_address_point"),

    CreatedBeforeScheduleRule(minimum_delta_hours=2.0),
]

validator = InputValidator(rules=rules)
valid_df, invalid_df, validation_report = validator.validate(input_df)

print(f'Valid entries: {(valid_df.head())}')
print(f'Invalid entries: {invalid_df.head()}')
print(f'Summary: {validation_report}')

# logger.info(
#     "Input validation completed",
#     extra=validation_report,
# )