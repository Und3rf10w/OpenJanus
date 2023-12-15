from time import sleep
from bs4 import BeautifulSoup
from cachetools import cached, TTLCache
import json
import re
import requests
from urllib.parse import urljoin
from typing import List, Dict, Any, Optional, Union
from openjanus.integrations.base import Integration


class ItemFinder(Integration):
    base_url: Optional[str] = "https://finder.cstone.space/"

    def __init__(self,
                 base_url: str = "https://finder.cstone.space/",
                 *args, 
                 **kwargs
        ):
        self.base_url = base_url
        self.session = requests.session()

    @cached(cache=TTLCache(maxsize=1024, ttl=3600))
    def search_items(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for items by name, returning a list of items that match the query. Input should be minimal, as the search is fuzzy.
        """
        response = self.session.get(urljoin(self.base_url, f"GetSearch"))
        response.raise_for_status()
        items = response.json()
        found_items = [item for item in items if query.lower() in item["name"].lower()]
        all_items = []
        if len(found_items) > 0 and len(found_items) <= 30:
            for item in found_items:
                item_details = self.get_item_details_and_data(item["id"])
                all_items.append(item_details)
                # Playing nice with cornerstone
                sleep(0.1)
        return all_items      
    

    @cached(cache=TTLCache(maxsize=1024, ttl=3600))
    def get_item_details(self, item_id: str) -> Union[str, None]:
        response = self.session.get(urljoin(self.base_url, f"Search/{item_id}"), allow_redirects=False)
        if response.status_code == 302:
            redirect_url = urljoin(self.base_url, response.headers["Location"])
            response = self.session.get(redirect_url)
            response.raise_for_status()
            return response.text
        else:
            response.raise_for_status()

    def parse_item_page(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML content of the item page to extract location and description data"""
        soup = BeautifulSoup(html_content, 'html.parser')
        location_data = []
        description = ""
        general_info = ""
        
        # Extract location data
        location_table = soup.find('table', id='table')
        if location_table:
            for row in location_table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                location_info = {
                    "location": cols[0].get_text(strip=True),
                    "base_price": cols[1].get_text(strip=True),
                    "verified": cols[2].get_text(strip=True) if len(cols) > 2 else ""
                }
                location_data.append(location_info)
        
        # Extract description
        pattern = re.compile(r"\s*DESCRIPTION", re.DOTALL)
        description_label = soup.find(text=pattern)
        if description_label:
            # Now we find the next div that could contain the description text
            # This is based on the assumption that the description follows in the next `div`
            description_content = description_label.find_next('div')
            if description_content:
                description = description_content.get_text(strip=True)

        # Extract general info

        general_label = soup.find(text="GENERAL")
        if general_label:
            general_table = general_label.find_next('table')
            if general_table is not None:
                general_data = []
                for row in general_table.find_all('tr'):
                    cols = row.find_all('td')
                    if len(cols) > 1:  # Skip rows without enough columns
                        row_data = {
                            cols[0].get_text(strip=True): cols[1].get_text(strip=True),
                        }
                        general_data.append(row_data)
                general_info = json.dumps(general_data)
                
        return {
            "general_info": general_info,
            "location_data": location_data,
            "description": description
        }

    def get_item_details_and_data(self, item_id: str) -> Dict[str, Any]:
        """
        Get the details of an item, including location and description data
        """
        item_page_html = self.get_item_details(item_id)
        if item_page_html:
            parsed_data = self.parse_item_page(item_page_html)
            return parsed_data
        return {"error": "Item details not found"}

