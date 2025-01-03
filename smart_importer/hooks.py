"""Importer decorators."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, Sequence

from beancount.core import data
from beangulp import Adapter, Importer, ImporterProtocol

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ImporterHook:
    """Interface for an importer hook."""

    def __call__(
        self,
        importer: Importer,
        file: str,
        imported_entries: data.Directives,
        existing: data.Directives,
    ) -> data.Directives:
        """Apply the hook and modify the imported entries.

        Args:
            importer: The importer that this hooks is being applied to.
            file: The file that is being imported.
            imported_entries: The current list of imported entries.
            existing: The existing entries, as passed to the extract
                function.

        Returns:
            The updated imported entries.
        """
        raise NotImplementedError


def apply_hooks(
    importer: Importer | ImporterProtocol,
    hooks: Sequence[
        Callable[
            [Importer, str, data.Directives, data.Directives], data.Directives
        ]
    ],
) -> Importer:
    """Apply a list of importer hooks to an importer.

    Args:
        importer: An importer instance.
        hooks: A list of hooks, each a callable object.
    """

    if not isinstance(importer, Importer):
        importer = Adapter(importer)
    unpatched_extract = importer.extract

    @wraps(unpatched_extract)
    def patched_extract_method(
        filepath: str, existing: data.Directives
    ) -> data.Directives:
        logger.debug("Calling the importer's extract method.")
        imported_entries = unpatched_extract(filepath, existing)

        for hook in hooks:
            imported_entries = hook(
                importer, filepath, imported_entries, existing
            )

        return imported_entries

    importer.extract = patched_extract_method
    importer.deduplicate = lambda entries, existing: None
    return importer
