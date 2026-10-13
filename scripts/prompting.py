"""
Shared prompt builder so training, evaluation and serving all format prompts identically.

- Old behaviour (no style): template.format(question=...)           -> prompts/sagan_style_template.txt
- New behaviour (style given): template also gets {audience}, {audience_desc} -> prompts/audience_style_template.txt
"""
from pathlib import Path

import yaml

STYLES_PATH = "prompts/audience_styles.yaml"


def load_styles(path=STYLES_PATH):
    with open(path) as f:
        return yaml.safe_load(f)["styles"]


def build_prompt(template, question, style=None, styles=None):
    if not style:
        return template.format(question=question)
    if styles is None:
        styles = load_styles()
    s = styles[style]
    return template.format(question=question, audience=s["name"], audience_desc=s["description"])


def load_template(path):
    return Path(path).read_text()
