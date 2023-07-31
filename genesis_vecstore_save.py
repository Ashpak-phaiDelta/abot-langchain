# WARNING! It does NOT clear previous data nor does it UPSERT. You need to manually
# clear the collection called 'genesis' in the vectorstore and then run this.


def _patch_langchain():
    from typing import Optional, List
    from langchain.vectorstores.pgvector import PGVector
    from sqlalchemy.orm import Session
    from sqlalchemy import delete

    def _delete_embeddings(self, ids: List[str] = None) -> None:
        with Session(self._conn) as session:
            query = delete(self.EmbeddingStore)
            if ids is not None:
                query = query.where(
                    self.EmbeddingStore.custom_id.in_(ids)
                )
            session.execute(query)
            session.commit()

    if not hasattr(PGVector, "delete_embeddings"):
        setattr(PGVector, "delete_embeddings", _delete_embeddings)


_patch_langchain()


from langchain.schema import Document
from pydantic import BaseModel, Field, Extra
from pydantic_yaml import to_yaml_str

import enum
import tqdm
import itertools
from datetime import timedelta
from urllib.parse import urljoin
from typing import Optional, List, Union

from requests_cache import CachedSession, SQLiteCache

from vectorstores.doc_chroma import chromadb
from vectorstores.genesis_pg import genesisdb
from genesis.config import GenesisSettings


_settings = GenesisSettings()

VECTORSTORE = genesisdb

GENESIS_BASE_URL = "https://api.phaidelta.com/backend"
GENESIS_TWC_WARLVL_ID = 10
CACHE_PATH = ".cache/requests.db"

REQUESTS_CACHE = SQLiteCache(db_path=CACHE_PATH, wal=True)
REQUESTS_CACHE_EXPIRY = timedelta(minutes=3)


class LiveServerSession(CachedSession):
    """Session with base url"""

    def __init__(self, base_url: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url

    def request(self, method, url, *args, **kwargs):
        url = urljoin(self.base_url.rstrip("/") + "/", url.lstrip("/"))
        return super().request(method, url, *args, **kwargs)


def make_batches(items: List, batch_size: int = 1) -> List:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# Call first time
# VECTORSTORE.create_collection()


class MakeDocsMixin:
    def _additional_metadata(self) -> dict:
        return {}

    def to_documents(self) -> List[Document]:
        return [
            Document(
                page_content=to_yaml_str(self),
                metadata={
                    "source": self._doc_source_template,
                    "content_type": "yaml",
                    **self._additional_metadata(),
                },
            )
        ]


class GenesisItemState(str, enum.Enum):
    NORMAL = "NORMAL"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    INACTIVE = "INACTIVE"


class GenesisLocationBase(BaseModel, extra=Extra.ignore):
    location_id: int
    location_name: str
    location_alias: Optional[str]


class GenesisUnitBase(BaseModel, extra=Extra.ignore):
    unit_id: int
    unit_name: str
    unit_alias: Optional[str]


class GenesisSensorSummary(
    MakeDocsMixin, BaseModel, extra=Extra.ignore, allow_population_by_field_name=True
):
    sensor_id: Optional[int]
    sensor_type: str
    sensor_subtype: str
    sensor_given_name: str
    sensor_value: Optional[str]
    sensor_measure_unit: Optional[str]
    sensor_health_state: GenesisItemState
    sensor_unit_at: GenesisUnitBase
    sensor_location_at: GenesisLocationBase
    sensor_type_alt: Optional[str]

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Sensor Summary: TWC/%s%s/%s sensors" % (
            self.sensor_location_at.location_name,
            ""
            if not self.sensor_location_at.location_alias
            else " (%s)" % self.sensor_location_at.location_alias,
            self.sensor_unit_at.unit_name
            if not self.sensor_unit_at.unit_alias
            else "%s" % self.sensor_unit_at.unit_alias,
        )

    def _additional_metadata(self) -> dict:
        return {
            "description": "Sensors in warehouses and units of TWC",  # inside warehouse location"# %s%s" % (
            #     self.sensor_location_at.location_name,
            #     '' if not self.sensor_location_at.location_alias else ' (%s)' % self.sensor_location_at.location_alias
            # ),
            "type": "genesis/sensor",
        }


class GenesisSensorSummaryWarehouse(GenesisSensorSummary):
    """Same as a regular sensor, but is designated as warehouse-level"""

    def _additional_metadata(self) -> dict:
        _old_metadata = super()._additional_metadata()
        _old_metadata.update({"subtype": "genesis/warehouse/sensor"})
        return _old_metadata


class GenesisUnitSummary(
    MakeDocsMixin, GenesisUnitBase, BaseModel, allow_population_by_field_name=True
):
    unit_sensors_out_count: Union[int, str]
    unit_health_state: GenesisItemState
    unit_location_at: GenesisLocationBase
    unit_sensors: List[GenesisSensorSummary] = Field(exclude=True)

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Warehouse-level Unit Summary: TWC/%s%s units" % (
            self.unit_location_at.location_name,
            ""
            if not self.unit_location_at.location_alias
            else " (%s)" % self.unit_location_at.location_alias,
        )

    def _additional_metadata(self) -> dict:
        return {
            "description": "Unit inside warehouse location %s%s"
            % (
                self.unit_location_at.location_name,
                ""
                if not self.unit_location_at.location_alias
                else " (%s)" % self.unit_location_at.location_alias,
            ),
            "type": "genesis/warehouse/unit",
        }

    def to_documents(self) -> List[Document]:
        return list(
            itertools.chain(
                super().to_documents(),
                *map(GenesisSensorSummary.to_documents, self.unit_sensors),
            )
        )


class GeoCoordinate(BaseModel):
    latitude: float
    longitude: float


class GenesisLocationSummaryItem(BaseModel):
    value: Optional[Union[int, str]]
    state: Optional[GenesisItemState]


class GenesisLocationSummaryPower(BaseModel):
    value: Optional[Union[float, int, str]]
    state: Optional[GenesisItemState]
    unit: Optional[str]


class GenesisLocationSummary(BaseModel, extra=Extra.allow):
    metrics: Optional[GenesisLocationSummaryItem]
    power: Optional[GenesisLocationSummaryPower]
    attendance: Optional[GenesisLocationSummaryItem]
    emergencies: Optional[GenesisLocationSummaryItem]


class GenesisLocation(
    MakeDocsMixin, GenesisLocationBase, BaseModel, allow_population_by_field_name=True
):
    location_coords: GeoCoordinate
    location_health_state: GenesisItemState
    location_summary: GenesisLocationSummary
    warehouse_units: List[GenesisUnitSummary] = Field(exclude=True)
    warehouse_sensors: List[GenesisSensorSummaryWarehouse] = Field(exclude=True)

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Warehouse Location: TWC/%s%s" % (
            self.location_name,
            "" if not self.location_alias else " (%s)" % self.location_alias,
        )

    def to_documents(self) -> List[Document]:
        return list(
            itertools.chain(
                super().to_documents(),
                *map(GenesisUnitSummary.to_documents, self.warehouse_units),
                *map(
                    GenesisSensorSummaryWarehouse.to_documents, self.warehouse_sensors
                ),
            )
        )

    def _additional_metadata(self) -> dict:
        return {
            "description": "Warehouse (location) used by company TWC",
            "type": "genesis/warehouse",
        }


class Genesis(BaseModel):
    class _GenesisMetadata(BaseModel, extra=Extra.allow):
        website_owner: str
        genesis_instance_owner: str

    locations: List[GenesisLocation]
    metadata: _GenesisMetadata

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Information: {genesis_instance_owner}"

    def to_documents(self) -> List[Document]:
        return [
            Document(
                page_content=to_yaml_str(self.metadata),
                metadata={
                    "source": self._doc_source_template.format(**self.metadata.dict()),
                    "content_type": "yaml",
                },
            ),
            *itertools.chain(*map(GenesisLocation.to_documents, self.locations)),
        ]


class DataCounts(MakeDocsMixin, BaseModel):
    all_warehouses: dict
    description: str
    warehouse: dict

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Warehouse Location: TWC - totals & number of warehouse, unit and sensor"


def genesis_counting(model: Genesis):
    return {
        "description": "All totals, counts of, total number of sensors at, units, warehouses, also grouped by state. Unit normal/inactive count, number of sensors",
        "all_warehouses": {
            "total": len(model.locations),
            "count_warehouses_by_state": {
                "count_normal": len(
                    list(
                        filter(
                            lambda x: x.location_health_state
                            == GenesisItemState.NORMAL,
                            model.locations,
                        )
                    )
                ),
                "count_out_of_range": len(
                    list(
                        filter(
                            lambda x: x.location_health_state
                            == GenesisItemState.OUT_OF_RANGE,
                            model.locations,
                        )
                    )
                ),
                "count_inactive": len(
                    list(
                        filter(
                            lambda x: x.location_health_state
                            == GenesisItemState.INACTIVE,
                            model.locations,
                        )
                    )
                ),
            },
        },
        "warehouse": {
            (
                warehouse.location_name
                + (
                    f" (warehouse.location_alias)"
                    if warehouse.location_alias is not None
                    else ""
                )
            ): {
                "total_warehouse_sensors": len(warehouse.warehouse_sensors),
                "total_warehouse_units": len(warehouse.warehouse_units),
                "total_sensors_in_warehouse_and_all_units": sum(
                    len(unit.unit_sensors) for unit in warehouse.warehouse_units
                )
                + len(warehouse.warehouse_sensors),
                "count_warehouse_sensors_by_state": {
                    "count_normal": len(
                        list(
                            filter(
                                lambda x: x.sensor_health_state
                                == GenesisItemState.NORMAL,
                                warehouse.warehouse_sensors,
                            )
                        )
                    ),
                    "count_out_of_range": len(
                        list(
                            filter(
                                lambda x: x.sensor_health_state
                                == GenesisItemState.OUT_OF_RANGE,
                                warehouse.warehouse_sensors,
                            )
                        )
                    ),
                    "count_inactive": len(
                        list(
                            filter(
                                lambda x: x.sensor_health_state
                                == GenesisItemState.INACTIVE,
                                warehouse.warehouse_sensors,
                            )
                        )
                    ),
                },
                "count_warehouse_units_by_state": {
                    "count_normal": len(
                        list(
                            filter(
                                lambda x: x.unit_health_state
                                == GenesisItemState.NORMAL,
                                warehouse.warehouse_units,
                            )
                        )
                    ),
                    "count_out_of_range": len(
                        list(
                            filter(
                                lambda x: x.unit_health_state
                                == GenesisItemState.OUT_OF_RANGE,
                                warehouse.warehouse_units,
                            )
                        )
                    ),
                    "count_inactive": len(
                        list(
                            filter(
                                lambda x: x.unit_health_state
                                == GenesisItemState.INACTIVE,
                                warehouse.warehouse_units,
                            )
                        )
                    ),
                },
            }
            for warehouse in model.locations
        },
    }


def scrape_all_genesis(sess: LiveServerSession) -> dict:
    """Sequential scrape of all data from APIs of Genesis"""
    raw_responses = {}

    try:
        locs = sess.get("/locations").json()

        raw_responses["locations"] = locs
        raw_responses["location_summary"] = {}  # Landing-level stuff
        raw_responses["warehouses"] = {}  # Warehouse-level stuff
        raw_responses["units"] = {}  # Unit-level stuff

        for loc in tqdm.tqdm(locs, unit="warehouse"):
            loc_id = loc["id"]
            loc_summary = sess.get(
                "/locations/{warehouse_id}/summary".format(warehouse_id=loc_id)
            ).json()
            raw_responses["location_summary"][loc_id] = loc_summary

            warlvl_details = sess.get(
                "/metrics/warehouse/{warehouse_id}".format(warehouse_id=loc_id)
            ).json()
            raw_responses["warehouses"][loc_id] = warlvl_details

            for unit in tqdm.tqdm(
                raw_responses["warehouses"][loc_id]["wv_unit_summary"], unit="unit"
            ):
                unit_id = unit["Unit Id"]
                unit_sensors = sess.get(
                    "/metrics/warehouse/{warehouse_id}/unit/{unit_id}".format(
                        warehouse_id=loc_id, unit_id=unit_id
                    )
                ).json()
                raw_responses["units"][unit_id] = unit_sensors
    finally:
        return raw_responses


def _get_alt_sensor_type(sensor) -> Optional[str]:
    if sensor["Metric Sub-Type"] == "RH":
        return "Humidity"
    if sensor["Metric Type"] == "Power - EM":
        return "Power/KWH/Energy Meter"


def parse_genesis_apis(responses: dict):
    parsed = {}

    def find_loc_alias(loc) -> Optional[str]:
        for unit in responses["warehouses"][loc["id"]]["wv_unit_summary"]:
            if unit["Location Alias"]:
                return unit["Location Alias"]

    parsed["metadata"] = {
        "website_owner": "phAIdelta",
        "genesis_instance_owner": "The Warehouse Company (TWC)",
    }

    parsed["locations"] = [
        {
            "location_id": loc["id"],
            "location_name": loc["name"],
            "location_coords": {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
            },
            "location_health_state": loc["state"],
            "location_alias": find_loc_alias(loc),
            "location_summary": responses["location_summary"][loc["id"]],
            "warehouse_sensors": [
                {
                    "sensor_id": sensor["Sensor Id"],
                    "sensor_type": sensor["Metric Type"],
                    "sensor_subtype": sensor["Metric Sub-Type"],
                    "sensor_given_name": sensor["Sensor Name"],
                    "sensor_value": sensor["Value"],
                    "sensor_measure_unit": sensor["Unit"],
                    "sensor_health_state": sensor["State"],
                    "sensor_unit_at": {
                        "unit_id": GENESIS_TWC_WARLVL_ID,
                        "unit_name": "Warehouse-level unit",
                        "unit_alias": "WARLVL (%s, %s)"
                        % (loc["name"], find_loc_alias(loc) or ""),
                    },
                    "sensor_location_at": {
                        "location_id": loc["id"],
                        "location_name": loc["name"],
                        "location_alias": find_loc_alias(loc),
                    },
                }
                for sensor in responses["warehouses"][loc["id"]]["wv_warehouse_metrics"]
            ],
            "warehouse_units": [
                {
                    "unit_id": unit["Unit Id"],
                    "unit_name": unit["Unit Name"],
                    "unit_alias": unit["Unit Alias"],
                    "unit_health_state": unit["State"],
                    "unit_sensors_out_count": unit["Value"],
                    "unit_location_at": {
                        "location_id": loc["id"],
                        "location_name": loc["name"],
                        "location_alias": find_loc_alias(loc),
                    },
                    "unit_sensors": [
                        {
                            "sensor_id": sensor["Sensor Id"],
                            "sensor_type": sensor["Metric Type"],
                            "sensor_subtype": sensor["Metric Sub-Type"],
                            "sensor_given_name": sensor["Sensor Name"],
                            "sensor_value": sensor["Value"],
                            "sensor_measure_unit": sensor["Unit"],
                            "sensor_health_state": sensor["State"],
                            "sensor_unit_at": {
                                "unit_id": unit["Unit Id"],
                                "unit_name": sensor["Unit Name"],
                                "unit_alias": sensor["Unit Alias"],
                            },
                            "sensor_location_at": {
                                "location_id": loc["id"],
                                "location_name": loc["name"],
                                "location_alias": find_loc_alias(loc),
                            },
                            "sensor_type_alt": _get_alt_sensor_type(sensor)
                            # "Sensor Name": "B2 Bsmnt 1_temp",
                            # "Sensor Alias": "B2 Bsmnt 1_temp",
                            # "Percentage": null,
                            # "Value Duration Minutes": null,
                            # "Threshold crosses": null
                        }
                        for sensor in responses["units"][unit["Unit Id"]][
                            "uv_unit_metrics"
                        ]
                    ],
                }
                for unit in responses["warehouses"][loc["id"]]["wv_unit_summary"]
            ],
        }
        for loc in responses["locations"]
    ]

    return parsed


if __name__ == "__main__":
    with LiveServerSession(
        base_url=GENESIS_BASE_URL,
        backend=REQUESTS_CACHE,
        expire_after=REQUESTS_CACHE_EXPIRY,
    ) as sess:
        docs_to_save: List[Document] = []
        sess.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % _settings.auth_token,
            }
        )

        import time

        start_time = time.time()
        print("Scraping APIs...")
        genesis_data = scrape_all_genesis(sess)

        print("Parsing...")
        parsed_data = parse_genesis_apis(genesis_data)
        genesis_twc = Genesis.parse_obj(parsed_data)
        docs_to_save.extend(genesis_twc.to_documents())

        print("Counting...")
        all_counting = genesis_counting(genesis_twc)
        all_data_counts = DataCounts.parse_obj(all_counting)
        docs_to_save.extend(all_data_counts.to_documents())

        print("Splitting...")
        # NOTE: Splitting JSON/YAML like this is a BAD IDEA!
        export_docs: List[
            Document
        ] = docs_to_save  # text_splitter.split_documents(docs_to_save)

        # Insert [Upsert] all documents

        print("Inserting...")
        insert_time = time.time()

        VECTORSTORE.delete_embeddings()

        for batch in tqdm.tqdm(
            make_batches(export_docs, batch_size=max(8, int(len(export_docs) / 25))),
            unit="doc",
        ):
            VECTORSTORE.add_documents(batch)

        if hasattr(VECTORSTORE, "persist"):
            VECTORSTORE.persist()
        finish_time = time.time()

        print(
            "Took %.2f seconds (%.2fs parsing, %.2fs uploading)"
            % (
                finish_time - start_time,
                insert_time - start_time,
                finish_time - insert_time,
            )
        )
