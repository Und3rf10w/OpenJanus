from time import sleep, time
from unicodedata import category
from bs4 import BeautifulSoup
from cachetools import cached, TTLCache
from cachetools.keys import hashkey
import json
import os
import re
import requests
from urllib.parse import urljoin
from typing import List, Dict, Any, Optional, Union
from openjanus.integrations.base import Integration


class PlanetarySurvey(Integration):
    base_url: str = "https://finder.cstone.space/"

    def __init__(self,
                 base_url: str = "https://finder.cstone.space/",
                 *args, 
                 **kwargs
        ):
        self.base_url = base_url
        self.session = requests.session()
        self.save_or_load_survey_data('survey_data.json')

    def _hashkey(self, params):
        return hashkey(params)
    

    def _search_helper(self, search_input: str):
        """
        Search for objects by name in a specific category, returning a list of items that even remotely match the query.
        The search is case-insensitive and partial matches are included.

        :param search_name: The name and category of the object to search for, comma delimited
        """
        location, category = [s.strip() for s in search_input.split(',')]
        return self.search_by_name(location, category)
    
    def download_and_save_survey_data(self, filename):
        """Download survey data and save it to a file."""
        self.survey_data = self._cornerstone_get_planetary_survey_data(f"api/surveyDataV2/*").json()
        with open(filename, 'w') as f:
            json.dump(self.survey_data, f)
    
    def save_or_load_survey_data(self, filename):
        """Save survey data to a file if it doesn't exist or is older than 6 hours. Otherwise, load it from the file."""
        # TODO: Actually do the stuff in the comment below
        six_hours_in_seconds = 6 * 60 * 60
        if os.path.exists(filename):
            file_age = time() - os.path.getmtime(filename)
            if file_age > six_hours_in_seconds:
                self.download_and_save_survey_data(filename)
            with open(filename, 'r') as f:
                self.survey_data = json.load(f)
        else:
            self.download_and_save_survey_data(filename)

    
    @cached(cache=TTLCache(maxsize=100000, ttl=3600), key=(_hashkey))
    def _cornerstone_get_planetary_survey_data(self, path: str, params: Optional[dict] = {"format": "json"}, allow_redirects: bool = True) -> requests.Response:
        response = self.session.get(urljoin(self.base_url, path), params=params, allow_redirects=allow_redirects)
        response.raise_for_status()
        return response


    def systems_info(self, system_name: str) -> List[Dict[str, Any]]:
        """
        Search for systems by name, returning a list of items that match the query. Input should be exact.
        """
        systems_data = self.survey_data['SystemsV2']
        return systems_data[system_name.capitalize()]
    
    
    def planet_info(self, planet_name: str) -> List[Dict[str, Any]]:
        """
        Search for planets by name, returning a list of items that match the query. Input should be exact.
        """
        planets_data = self.survey_data['PlanetsV2']
        return planets_data[planet_name.capitalize()]
    
    def location_info(self, location_name: str) -> List[Dict[str, Any]]:
        """
        Search for locations by name, returning a list of items that match the query. Input should be exact.
        """
        locations_data = self.survey_data['LocationsV2']
        return locations_data[location_name.capitalize()]
    

    @cached(cache=TTLCache(maxsize=1024, ttl=3600))
    def search_by_name(self, search_name: str, category: str) -> List[Dict[str, Any]]:
        """
        Search for objects by name in a specific category, returning a list of items that even remotely match the query.
        The search is case-insensitive and partial matches are included.

        :param search_name: The name of the object to search for
        :param category: The category to search in ('SystemsV2', 'PlanetsV2', 'LocationsV2')
        :return: A list of objects that match the search query
        """
        if category not in ['SystemsV2', 'PlanetsV2', 'LocationsV2']:
            raise ValueError(f"Invalid category {category}")

        search_name_lower = search_name.lower()
        data = self.survey_data[category]

        # Use regex to allow for partial, case-insensitive matches
        pattern = re.compile(re.escape(search_name_lower), re.IGNORECASE)

        # Find and return all matches
        matches = [
            item for item in data
            if any(pattern.search(str(value).lower()) for key, value in item.items() if isinstance(value, str))
        ]
        return matches