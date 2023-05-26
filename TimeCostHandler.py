import logging
from typing import List, Dict, Any
from llama_index.callbacks.schema import CBEventType
from llama_index.callbacks.base import BaseCallbackHandler
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseEvent:
    def __init__(
        self,
        event_type: CBEventType,
        start_time: str
        ) -> None:
        self.event_type = event_type
        self.start_time = start_time
    def count_cost(
        self,
        cur_time:str
        ) -> str:
        cost = datetime.strptime(cur_time,'%b-%d-%Y %H:%M:%S') - datetime.strptime(self.start_time,'%b-%d-%Y %H:%M:%S')
        return cost.total_seconds()

class TimeCostHandler(BaseCallbackHandler):
    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType],
        event_ends_to_ignore: List[CBEventType]
    ) -> None:
        super().__init__(event_starts_to_ignore, event_ends_to_ignore)
        self.event_id_dict = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Dict[str, Any] | None = None,
        event_id: str = "", **kwargs: Any
    ) -> str:
        cur_time = datetime.now().strftime('%b-%d-%Y %H:%M:%S')
        logger.info("begin eventType:%s eventId:%s start_time:%s" %
                    (event_type, event_id, cur_time))
        if event_id not in self.event_id_dict:
            self.event_id_dict[event_id] = []
        self.event_id_dict[event_id].append(BaseEvent(event_type,cur_time))

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Dict[str, Any] | None = None,
        event_id: str = "", **kwargs: Any
    ) -> None:
        cur_time = datetime.now().strftime('%b-%d-%Y %H:%M:%S')
        end_list = self.event_id_dict[event_id]
        for x in end_list:
            logger.info("end eventType:%s eventId:%s end_time:%s cost_time:%ss" %
                        (x.event_type, event_id, cur_time, x.count_cost(cur_time)))
        self.event_id_dict.pop(event_id)
