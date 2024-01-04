from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import logging
from functools import lru_cache
from datetime import datetime
import ahocorasick
import pickle
import os
from spell_checker import SpellChecker


# TODO: Correctly register your component with its type
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER], is_trainable=False
)
class CustomNLUComponent(GraphComponent):
    def __init__(self) -> None:
        super().__init__()

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # TODO: This is the method which Rasa Open Source will call during inference.
        """Processes incoming messages and computes and sets features."""
        spell_checker = SpellChecker()
        for message in messages:
            logging.info(
                f"############### before: {message.data['text']} ############### "
            )
            msg_text = message.data["text"]
            edited_text = spell_checker.spell(msg_text)
            logging.info(f"############### after: {edited_text} ############### ")
            message.data["text"] = edited_text

        return messages
