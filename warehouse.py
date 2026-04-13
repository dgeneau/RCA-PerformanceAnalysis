# warehouse_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from datetime import datetime, date
import requests
from jsonschema import validate as jsonschema_validate, ValidationError as JSONSchemaValidationError


class WarehouseClientError(Exception):
    """Raised for configuration, network, and server-errors from the Warehouse client."""


@dataclass
class WarehouseAPIConfig:
    """
    Minimal configuration for the Warehouse API.
    """
    base_url: str                                  # e.g. "https://example.com"
    ingest_path: str                = "/api/warehouse/ingestion/primary/"          # POST for raw ingestion
    datasource_detail_pattern: str  = "/api/warehouse/data-sources/{uuid}/"  # GET for ds details
    records_list_path: str          = "/api/warehouse/data-records/"  # GET list w/ filters
    timeout_s: int = 20

    @property
    def ingest_url(self) -> str:
        return self.base_url.rstrip("/") + self.ingest_path

    def datasource_url(self, ds_uuid: str) -> str:
        return self.base_url.rstrip("/") + self.datasource_detail_pattern.format(uuid=ds_uuid)

    @property
    def records_url(self) -> str:
        return self.base_url.rstrip("/") + self.records_list_path

class WarehouseClient:
    """
    Generic client for interacting with the Warehouse ingestion endpoint(s).
    - No assumptions about record shape or field names
    - Optional client-side validation against the head schema for any stream (default 'raw')
    """

    def __init__(self, config: WarehouseAPIConfig, token_getter: Callable[[], str]):
        """
        token_getter: callable that returns an OAuth access token (without the 'Bearer ' prefix).
        """
        if not callable(token_getter):
            raise TypeError("token_getter must be callable and return a token string.")
        self.config = config
        self._token_getter = token_getter
        self._schema_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (source_uuid, stream) -> schema

    # ---------------------------------------------------------------------
    # Public: schema helpers
    # ---------------------------------------------------------------------
    def get_head_schema(self, *, source_uuid: str, stream: str = "primary", force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch and cache the head JSON Schema for a given (source_uuid, stream).
        Returns the schema dict, or None if unavailable.
        """
        cache_key = (source_uuid, stream)
        if not force_refresh and cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        # Currently the server exposes head schema for PRIMARY on DataSource detail as 'head_primary_definition'
        # If/when additional streams are surfaced, you can extend this logic.
        url = self.config.datasource_url(source_uuid)
        headers = {"Authorization": self._bearer(), "Accept": "application/json"}
        try:
            resp = requests.get(url, headers=headers, timeout=self.config.timeout_s)
        except requests.RequestException as e:
            # Schema is optional; don't fail hard
            return None

        if not resp.ok:
            return None

        try:
            data = resp.json()
        except ValueError:
            return None

        if stream == "primary":
            head_def = data.get("head_raw_definition") or {}
            schema = head_def.get("schema")
        else:
            # Future: if server exposes e.g. head_{stream}_definition
            schema = None

        if isinstance(schema, dict):
            self._schema_cache[cache_key] = schema
            return schema
        return None

    def validate_records(self, *, records: Iterable[Dict[str, Any]], schema: Dict[str, Any]) -> None:
        """
        Validate an iterable of records against a provided JSON Schema.
        Raises WarehouseClientError on first failure.
        """
        for i, rec in enumerate(records):
            try:
                jsonschema_validate(rec, schema)
            except JSONSchemaValidationError as e:
                raise WarehouseClientError(f"Record {i} failed schema validation: {e.message}") from e

    def validate_against_head_schema(self, *, source_uuid: str, records: Iterable[Dict[str, Any]], stream: str = "primary") -> None:
        """
        Convenience: fetch head schema for (source_uuid, stream) and validate records against it.
        Raises WarehouseClientError if schema is unavailable or validation fails.
        """
        schema = self.get_head_schema(source_uuid=source_uuid, stream=stream)
        if not schema:
            raise WarehouseClientError(f"No head schema available for stream '{stream}' on source {source_uuid}.")
        self.validate_records(records=records, schema=schema)

    # ---------------------------------------------------------------------
    # Public: ingestion
    # ---------------------------------------------------------------------
    def ingest_raw(
        self,
        *,
        source_uuid: str,
        records: List[Dict[str, Any]],
        subject_field: Optional[str] = None,
        validate_client_side: bool = False,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Generic ingestion into the PRIMARY endpoint.

        Params:
            source_uuid: UUID of DataSource to ingest into.
            records: list of dict rows to store as DataRecords.
            subject_field: optional field name in each record that contains a Profile PK.
            validate_client_side: when True, validates records against the head 'primary' schema first.

        Returns:
            (dataset_dict, created_records_count)

        Raises:
            WarehouseClientError on failures.
        """
        if not records:
            raise WarehouseClientError("No records provided.")

        if validate_client_side:
            self.validate_against_head_schema(source_uuid=source_uuid, records=records, stream="primary")

        payload = {
            "source": source_uuid,
            "records": records,
        }
        if subject_field:
            payload["subject_field"] = subject_field

        headers = {
            "Authorization": self._bearer(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            resp = requests.post(
                self.config.ingest_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.config.timeout_s,
            )
        except requests.RequestException as e:
            raise WarehouseClientError(f"Network error posting to ingest endpoint: {e}") from e

        if resp.status_code == 201:
            try:
                body = resp.json()
            except ValueError:
                raise WarehouseClientError("Server returned 201 but response body was not valid JSON.")

            dataset = body.get("dataset")
            created = int(body.get("created_records", 0))
            if not dataset or "uuid" not in dataset:
                raise WarehouseClientError("Ingest succeeded but response missing dataset info.")
            return dataset, created

        # Surface server error details if present
        try:
            err = resp.json()
            detail = err.get("detail") or err
        except ValueError:
            detail = resp.text or f"HTTP {resp.status_code}"
        raise WarehouseClientError(f"Ingest failed: {detail}")

    def list_records(
            self,
            *,
            source_uuid: str,
            collected_after: Optional[Union[str, datetime, date]] = None,
            collected_before: Optional[Union[str, datetime, date]] = None,
            role: Optional[str] = None,  # tip: omit to use server default (primary)
            subject: Optional[int] = None,
            page_size: Optional[int] = None,
            extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch ALL records that match the filters, following pagination until exhausted.
        Use `iter_records` for streaming results instead of building a full list.
        """
        results: List[Dict[str, Any]] = []
        for rec in self.iter_records(
                source_uuid=source_uuid,
                collected_after=collected_after,
                collected_before=collected_before,
                role=role,
                subject=subject,
                page_size=page_size,
                extra_params=extra_params,
        ):
            results.append(rec)
        return results

    def iter_records(
            self,
            *,
            source_uuid: str,
            collected_after: Optional[Union[str, datetime, date]] = None,
            collected_before: Optional[Union[str, datetime, date]] = None,
            role: Optional[str] = None,  # tip: omit to use server default (primary)
            subject: Optional[int] = None,
            page_size: Optional[int] = None,
            extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Lazily iterate over matching records. Follows DRF pagination via `next` links.
        """
        params: Dict[str, Any] = {
            "source_uuid": source_uuid,
        }
        if collected_after is not None:
            params["collected_after"] = self._to_iso8601(collected_after)
        if collected_before is not None:
            params["collected_before"] = self._to_iso8601(collected_before)
        if role:
            params["role"] = role  # if omitted, server defaults to primary
        if subject is not None:
            params["subject"] = int(subject)
        if page_size is not None:
            # works if your API uses PageNumberPagination; ignored otherwise
            params["limit"] = int(page_size)

        if extra_params:
            params.update(extra_params)

        url = self.config.records_url
        while url:
            payload = self._GET_json(url, params=params)
            # After first request, subsequent `next` URLs already include query; don't resend params.
            params = None

            if isinstance(payload, dict) and "results" in payload:
                # paginated
                items = payload.get("results") or []
                for item in items:
                    yield item
                url = payload.get("next")
            elif isinstance(payload, list):
                # unpaginated list
                for item in payload:
                    yield item
                url = None
            else:
                # unexpected shape
                raise WarehouseClientError("Unexpected response from records endpoint.")


    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _bearer(self) -> str:
        token = self._token_getter()
        if not token:
            raise WarehouseClientError("No access token available.")
        return token if token.lower().startswith("bearer ") else f"Bearer {token}"

    def _GET_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        headers = {
            "Authorization": self._bearer(),
            "Accept": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=self.config.timeout_s)
        except requests.RequestException as e:
            raise WarehouseClientError(f"GET {url} failed: {e}") from e

        if not resp.ok:
            try:
                err = resp.json()
                detail = err.get("detail") or err
            except ValueError:
                detail = resp.text or f"HTTP {resp.status_code}"
            raise WarehouseClientError(f"GET {url} failed: {detail}")

        try:
            return resp.json()
        except ValueError as e:
            raise WarehouseClientError("Response was not valid JSON.") from e

    @staticmethod
    def _to_iso8601(dt: Union[str, datetime, date]) -> str:
        """
        Coerce a date/datetime/ISO string into an ISO-8601 string acceptable by DRF filters.
        If naive datetime is passed, treat it as UTC.
        """
        if isinstance(dt, str):
            return dt
        if isinstance(dt, date) and not isinstance(dt, datetime):
            # Convert date -> midnight UTC
            return datetime(dt.year, dt.month, dt.day).isoformat() + "Z"
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.isoformat() + "Z"
            return dt.isoformat()
        raise TypeError("collected_after/collected_before must be str, date, or datetime")