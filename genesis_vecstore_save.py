
# WARNING! It does NOT clear previous data nor does it UPSERT. You need to manually
# clear the collection called 'genesis' in the vectorstore and then run this.


from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document
from pydantic import BaseModel, Field, Extra, parse_obj_as

import enum
import tqdm
from mezmorize import Cache
import requests
import itertools
from urllib.parse import urljoin
from typing import Optional, List, Union
from typing_extensions import NotRequired

from vectorstores.doc_chroma import chromadb
from vectorstores.genesis_pg import genesisdb


GENESIS_BASE_URL = "https://api.phaidelta.com/backend"
GENESIS_AUTH_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZF91c2VyIjoxMywiaWF0IjoxNjg3NzU3ODU4fQ.14AjrkViISuzUZMrW2WJlAKLxhDgWHIJFWYkhBrkaLQ'

SPLIT_CHUNK_SIZE = 500
SPLIT_CHUNK_OVERLAP = 30


request_cache = Cache(CACHE_TYPE='filesystem', CACHE_DIR=".cache/")


class LiveServerSession(requests.Session):
    '''Session with base url'''
    def __init__(self, base_url: Optional[str] = None):
        super().__init__()
        self.base_url = base_url

    @request_cache.memoize(500)
    def request(self, method, url, *args, **kwargs):
        url = urljoin(self.base_url.rstrip("/") + "/", url.lstrip("/"))
        return super().request(method, url, *args, **kwargs)

def make_batches(items: List, batch_size: int = 1) -> List:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SPLIT_CHUNK_SIZE,
    chunk_overlap=SPLIT_CHUNK_OVERLAP
)

# Call first time
# genesisdb.create_collection()


class MakeDocsMixin:
    def _additional_metadata(self) -> dict:
        return {}

    def to_documents(self) -> List[Document]:
        return [Document(
            page_content=self.json(),
            metadata={
                "source": self._doc_source_template,
                "content_type": "json",
                **self._additional_metadata()
            }
        )]


class GenesisItemState(str, enum.Enum):
    NORMAL = "NORMAL"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    INACTIVE = "INACTIVE"

class GenesisLocationBase(BaseModel):
    # location_id: int
    location_name: str
    location_alias: Optional[str]

class GenesisUnitBase(BaseModel):
    # unit_id: int
    unit_name: str
    unit_alias: Optional[str]


class GenesisSensorSummary(MakeDocsMixin, BaseModel):
    # sensor_id: Optional[int]
    sensor_type: str
    sensor_subtype: str
    sensor_given_name: str
    sensor_value: Optional[str]
    sensor_measure_unit: Optional[str]
    sensor_health_state: GenesisItemState
    sensor_unit_at: GenesisUnitBase
    sensor_location_at: GenesisLocationBase

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Sensor Summary: TWC/%s%s/%s sensors" % (
            self.sensor_location_at.location_name,
            '' if not self.sensor_location_at.location_alias else ' (%s)' % self.sensor_location_at.location_alias,

            self.sensor_unit_at.unit_name if not self.sensor_unit_at.unit_alias else '%s' % self.sensor_unit_at.unit_alias
        )

    def _additional_metadata(self) -> dict:
        return {
            "description": "Sensors in warehouses and units of TWC" # inside warehouse location"# %s%s" % (
            #     self.sensor_location_at.location_name,
            #     '' if not self.sensor_location_at.location_alias else ' (%s)' % self.sensor_location_at.location_alias
            # )
        }

    class Config:
        allow_population_by_field_name = True


class GenesisUnitSummary(MakeDocsMixin, GenesisUnitBase, BaseModel):
    unit_sensors_out_count: Union[int, str]
    unit_health_state: GenesisItemState
    unit_location_at: GenesisLocationBase
    unit_sensors: List[GenesisSensorSummary] = Field(exclude=True)

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Warehouse-level Unit Summary: TWC/%s%s units" % (
            self.unit_location_at.location_name,
            '' if not self.unit_location_at.location_alias else ' (%s)' % self.unit_location_at.location_alias,
        )

    def _additional_metadata(self) -> dict:
        return {
            "description": "Unit inside warehouse location %s%s" % (
                self.unit_location_at.location_name,
                '' if not self.unit_location_at.location_alias else ' (%s)' % self.unit_location_at.location_alias
            )
        }

    def to_documents(self) -> List[Document]:
        return list(itertools.chain(
            super().to_documents(),
            *map(GenesisSensorSummary.to_documents, self.unit_sensors)
        ))

    class Config:
        allow_population_by_field_name = True


class GeoCoordinate(BaseModel):
    latitude: float
    longitude: float


class GenesisLocationSummaryItem(BaseModel):
    value: Optional[Union[float, int, str]]
    state: Optional[GenesisItemState]


class GenesisLocationSummaryPower(GenesisLocationSummaryItem, BaseModel):
    unit: Optional[str]


class GenesisLocationSummary(BaseModel):
    metrics: Optional[GenesisLocationSummaryItem]
    power: Optional[GenesisLocationSummaryPower]
    attendance: Optional[GenesisLocationSummaryItem]
    emergencies: Optional[GenesisLocationSummaryItem]

    class Config:
        extra = Extra.allow


class GenesisLocation(MakeDocsMixin, GenesisLocationBase, BaseModel):
    location_coords: GeoCoordinate
    location_health_state: GenesisItemState
    location_summary: GenesisLocationSummary
    warehouse_units: List[GenesisUnitSummary] = Field(exclude=True)
    warehouse_sensors: List[GenesisSensorSummary] = Field(exclude=True)

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Warehouse Location: TWC/%s%s" % (
            self.location_name,
            '' if not self.location_alias else ' (%s)' % self.location_alias
        )

    def to_documents(self) -> List[Document]:
        return list(itertools.chain(
            super().to_documents(),
            *map(GenesisUnitSummary.to_documents, self.warehouse_units),
            *map(GenesisSensorSummary.to_documents, self.warehouse_sensors)
        ))

    def _additional_metadata(self) -> dict:
        return {
            "description": "Warehouse (location) used by company TWC"
        }

    class Config:
        allow_population_by_field_name = True


class Genesis(BaseModel):
    class _GenesisMetadata(BaseModel):
        website_owner: str
        genesis_instance_owner: str
        class Config:
            extra = Extra.allow
    locations: List[GenesisLocation]
    metadata: _GenesisMetadata

    @property
    def _doc_source_template(self) -> str:
        return "Genesis Information: {genesis_instance_owner}"

    def to_documents(self) -> List[Document]:
        return [
            Document(
                page_content=self.metadata.json(),
                metadata={
                    "source": self._doc_source_template.format(**self.metadata.dict()),
                    "content_type": "json"
                }
            ),
            *itertools.chain(*map(GenesisLocation.to_documents, self.locations))
        ]


def scrape_all_genesis(sess: LiveServerSession) -> dict:
    '''Sequential scrape of all data from APIs of Genesis'''
    response = {}
    raw_responses = {}
    parsed = {}
    response['raw_responses'] = raw_responses
    response['parsed'] = parsed
    try:
        locs = sess.get('/locations').json()
        raw_responses['locations'] = locs
        raw_responses['location_summary'] = {}
        raw_responses['warehouses'] = {} # Warehouse-level stuff
        raw_responses['units'] = {} # Unit-level stuff

        for loc in tqdm.tqdm(locs):
            loc_id = loc['id']
            loc_summary = sess.get("/locations/{warehouse_id}/summary".format(warehouse_id=loc_id)).json()
            raw_responses['location_summary'][loc_id] = loc_summary

            warlvl_details = sess.get("/metrics/warehouse/{warehouse_id}".format(warehouse_id=loc_id)).json()
            raw_responses['warehouses'][loc_id] = warlvl_details

            for unit in tqdm.tqdm(raw_responses['warehouses'][loc_id]['wv_unit_summary']):
                unit_id = unit['Unit Id']
                unit_sensors = sess.get("/metrics/warehouse/{warehouse_id}/unit/{unit_id}".format(warehouse_id=loc_id, unit_id=unit_id)).json()
                raw_responses['units'][unit_id] = unit_sensors

    finally:
        def find_loc_alias(loc) -> Optional[str]:
            for unit in raw_responses['warehouses'][loc['id']]['wv_unit_summary']:
                if unit['Location Alias']:
                    return unit['Location Alias']
        parsed['metadata'] = {
            'website_owner': 'phAIdelta',
            'genesis_instance_owner': 'The Warehouse Company (TWC)'
        }
        parsed['locations'] = [
            {
                'location_id': loc['id'],
                'location_name': loc['name'],
                'location_coords': {
                    'latitude': loc['latitude'],
                    'longitude': loc['longitude']
                },
                'location_health_state': loc['state'],
                'location_alias': find_loc_alias(loc),
                "location_summary": raw_responses['location_summary'][loc['id']],
                "warehouse_sensors": [
                    {
                        # 'sensor_id': sensor['Sensor Id'],
                        'sensor_type': sensor['Metric Type'],
                        'sensor_subtype': sensor['Metric Sub-Type'],
                        'sensor_given_name': sensor['Sensor Name'],
                        'sensor_value': sensor['Value'],
                        'sensor_measure_unit': sensor['Unit'],
                        'sensor_health_state': sensor['State'],
                        'sensor_unit_at': {
                            # 'unit_id': -1,
                            'unit_name': 'Warehouse-level unit',
                            'unit_alias': 'WARLVL / %s / %s' % (loc['name'], find_loc_alias(loc) or '')
                        },
                        'sensor_location_at': {
                            # 'location_id': loc['id'],
                            'location_name': loc['name'],
                            'location_alias': find_loc_alias(loc)
                        }
                    }
                    for sensor in raw_responses['warehouses'][loc['id']]['wv_warehouse_metrics']
                ],
                "warehouse_units": [
                    {
                        # 'unit_id': unit['Unit Id'],
                        'unit_name': unit['Unit Name'],
                        'unit_alias': unit['Unit Alias'],
                        'unit_health_state': unit['State'],
                        'unit_sensors_out_count': unit['Value'],
                        'unit_location_at': {
                            # 'location_id': loc['id'],
                            'location_name': loc['name'],
                            'location_alias': find_loc_alias(loc)
                        },
                        'unit_sensors': [
                            {
                                # 'sensor_id': '',
                                'sensor_type': sensor['Metric Type'],
                                'sensor_subtype': sensor['Metric Sub-Type'],
                                'sensor_given_name': sensor['Sensor Name'],
                                'sensor_value': sensor['Value'],
                                'sensor_measure_unit': sensor['Unit'],
                                'sensor_health_state': sensor['State'],

                                'sensor_unit_at': {
                                    # 'unit_id': -1,
                                    'unit_name': sensor['Unit Name'],
                                    'unit_alias': sensor['Unit Alias']
                                },
                                'sensor_location_at': {
                                    # 'location_id': loc['id'],
                                    'location_name': loc['name'],
                                    'location_alias': find_loc_alias(loc)
                                }

                                # "Sensor Name": "B2 Bsmnt 1_temp",
                                # "Sensor Alias": "B2 Bsmnt 1_temp",
                                # "Percentage": null,
                                # "Value Duration Minutes": null,
                                # "Threshold crosses": null
                            }
                            for sensor in raw_responses['units'][unit['Unit Id']]['uv_unit_metrics']
                        ]
                    }
                    for unit in raw_responses['warehouses'][loc['id']]['wv_unit_summary']
                ]
            } for loc in raw_responses['locations']
        ]
        return response


if __name__ == "__main__":
    with LiveServerSession(base_url=GENESIS_BASE_URL) as sess:
        docs_to_save: List[Document] = []
        sess.headers.update({
            'Content-Type':'application/json',
            'Authorization': 'Bearer %s' % GENESIS_AUTH_TOKEN
        })

        import time
        start_time = time.time()
        print("Scraping APIs...")
        data = scrape_all_genesis(sess)
        # print("Model:", data['parsed'])
        print("Parsing...")
        genesis_twc = Genesis.parse_obj(data['parsed'])
        docs_to_save.extend(genesis_twc.to_documents())

        print("Splitting...")
        export_docs: List[Document] = text_splitter.split_documents(docs_to_save)

        # Insert all documents
        # genesisdb.delete()

        print("Inserting...")
        for batch in tqdm.tqdm(make_batches(export_docs, batch_size=max(8, int(len(export_docs)/100)))):
            genesisdb.add_documents(batch)
        # genesisdb.persist()

        print("Took %.2f seconds" % (time.time() - start_time))
