"""Importer decorators."""
import logging
from functools import wraps

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ImporterHook:
    """Interface for an importer hook."""

    def __call__(self, importer, file, imported_entries, existing_entries):
        """Apply the hook and modify the imported entries.

        Args:
            importer: The importer that this hooks is being applied to.
            file: The file that is being imported.
            imported_entries: The current list of imported entries.
            existing_entries: The existing entries, as passed to the extract
                function.

        Returns:
            The updated imported entries.
        """
        raise NotImplementedError


def apply_hooks(importer, hooks):
    """Apply a list of importer hooks to an importer.

    Args:
        importer: An importer instance.
        hooks: A list of hooks, each a callable object.
    """

    unpatched_extract = importer.extract

    @wraps(unpatched_extract)
    def patched_extract_method(file, existing_entries=None):
        logger.debug("Calling the importer's extract method.")
        imported_entries = unpatched_extract(
            file, existing_entries=existing_entries
        )

        for hook in hooks:
            imported_entries = hook(
                importer, file, imported_entries, existing_entries
            )

        return imported_entries

    importer.extract = patched_extract_method
    return importer
