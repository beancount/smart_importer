"""Wrap importers with smart_importer predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from beangulp.importer import Importer

if TYPE_CHECKING:
    import datetime

    from beancount.core.data import Directive

    from smart_importer.predictor import EntryPredictor


class ImporterWrapper(Importer):
    """Wrapper around an importer for enriching it with smart importer logic.

    Args:
        importer: The importer to wrap
        predictor: The entry predictor
    """

    def __init__(self, importer: Importer, predictor: EntryPredictor) -> None:
        self.importer = importer
        self.predictor = predictor

    @property
    def name(self) -> str:
        return self.importer.name

    def identify(self, filepath: str) -> bool:
        return self.importer.identify(filepath)

    def account(self, filepath: str) -> str:
        return self.importer.account(filepath)

    def date(self, filepath: str) -> datetime.date | None:
        return self.importer.date(filepath)

    def filename(self, filepath: str) -> str | None:
        return self.importer.filename(filepath)

    def deduplicate(
        self, entries: list[Directive], existing: list[Directive]
    ) -> None:
        return self.importer.deduplicate(entries, existing)

    def sort(self, entries: list[Directive], reverse: bool = False) -> None:
        return self.importer.sort(entries, reverse)

    def extract(
        self, filepath: str, existing: list[Directive]
    ) -> list[Directive]:
        entries = self.importer.extract(filepath, existing)
        account = self.importer.account(filepath)
        modified_entries = self.predictor.hook(
            [(filepath, entries, account, self.importer)], existing
        )
        return modified_entries[0][1]
