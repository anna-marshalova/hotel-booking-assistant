import datetime
import random

from num2words import num2words

from src.constants import RANDOM_SEED

random.seed(RANDOM_SEED)

DEFAULT_DATE_FORMAT = "%Y/%m/%d"


DATE_FORMATS_WITH_YEAR = [
    "%Y/%m/%d",
    "%y/%m/%d",
    "%d/%m/%y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%y-%m-%d",
    "%d.%m.%y",
    "%d.%m.%Y",
    "%d-%m-%y",
    "%d-%m-%Y",
    "%Y %B %d",
    "%Y %B-%d",
    "%B %d %Y",
    "%b %d %Y",
    "the %B of %d %Y",
]
DATE_FORMATS_WITHOUT_YEAR = [
    "%m/%d",
    "%m-%d",
    "%d.%m",
    "%B %d",
    "%b %d",
    "%d %B",
    "%d %b",
    "the %d of %B",
]

END_DATE_VARIANTS = ["tomorrow", "day after tomorrow", "{date}", "next {weekday}"]
START_DATE_VARIANTS = ["today", "on {date}", *END_DATE_VARIANTS]
PERIOD_VARIANTS = [
    "{n} days",
    "{n} nights",
    "{n} weeks",
    "a week",
    "one week",
    "1 week",
    "a night",
    "one night",
    "1 night",
    "one day",
    "1 day",
    "a day",
]


def random_today(start_year=2020, end_year=2030):
    start = datetime.date(day=1, month=1, year=start_year)
    end = datetime.date(day=31, month=12, year=end_year)
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + datetime.timedelta(seconds=random_second)


def random_format_date(date, year_prob=0.3):
    if random.random() < year_prob:
        format = random.choice(DATE_FORMATS_WITH_YEAR)
    else:
        format = random.choice(DATE_FORMATS_WITHOUT_YEAR)
    return date.strftime(format)


def random_dates(today, max_delta=100):
    start_delta = random.randint(0, max_delta - 1)
    end_delta = random.randint(start_delta, max_delta)
    return today + datetime.timedelta(days=start_delta), today + datetime.timedelta(
        days=end_delta
    )


def random_period(max_days=30, max_weeks=4, to_word_prob=0.3):
    period_template = random.choice(PERIOD_VARIANTS)
    if "{n}" in period_template:
        if "week" in period_template:
            n = random.randint(2, max_weeks)
        else:
            n = random.randint(2, max_days)
        if random.random() < to_word_prob:
            n_str = num2words(n)
        else:
            n_str = str(n)
        period = period_template.format(n=n_str)
    else:
        n = 1
        period = period_template
    delta = period2timedelta(period, n)
    return period, delta


def period2timedelta(period, n):
    if "week" in period:
        return datetime.timedelta(days=n * 7)
    return datetime.timedelta(days=n)


def get_date(date_template, today, date):
    if date_template == "today":
        return today
    if date_template == "tomorrow":
        return today + datetime.timedelta(days=1)
    return date


def format_date(date_template, date):
    date_str = random_format_date(date)
    return date_template.format(weekday=date.strftime("%A"), date=date_str)


def get_stay_dates(today):
    end_date_template = random.choice(END_DATE_VARIANTS)
    if end_date_template == "tomorrow":
        start_date_template = "today"
    elif end_date_template == "day after tomorrow":
        start_date_template = random.choice(["today", "tomorrow"])
    else:
        start_date_template = random.choice(START_DATE_VARIANTS)
    if "weekday" in start_date_template or "weekday" in end_date_template:
        start_date, end_date = random_dates(today, max_delta=7)
    else:
        start_date, end_date = random_dates(today)
    start_date = get_date(start_date_template, today, start_date)
    end_date = get_date(end_date_template, today, end_date)
    start_date_str = format_date(start_date_template, start_date)
    end_date_str = format_date(end_date_template, end_date)
    return start_date, end_date, start_date_str, end_date_str


def get_user_date_dict(user_message_template, start_date, end_date, period_delta):
    date_output = {
        "start_date": start_date.strftime(DEFAULT_DATE_FORMAT),
        "end_date": "",
    }
    if "end_date" in user_message_template:
        date_output["end_date"] = end_date.strftime(DEFAULT_DATE_FORMAT)
    elif "stay_period" in user_message_template:
        date_output["end_date"] = (start_date + period_delta).strftime(
            DEFAULT_DATE_FORMAT
        )
    return date_output
