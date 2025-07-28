# anna_archive.py
import requests
from bs4 import BeautifulSoup
import re
import logging
from typing import List, Dict, Optional
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnnaArchiveAPI:
    def __init__(self):
        self.base_url = "https://annas-archive.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.rate_limit_delay = 1  # 1 second between requests

    def _make_request(self, url: str, params: dict = None, timeout: int = 15) -> Optional[requests.Response]:
        """Make a rate-limited request with error handling"""
        try:
            # Add random delay to avoid being blocked
            time.sleep(self.rate_limit_delay + random.uniform(0, 0.5))
            
            response = self.session.get(url, params=params, timeout=timeout)
            
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limited, waiting longer...")
                time.sleep(5)
                response = self.session.get(url, params=params, timeout=timeout)
            
            return response if response.status_code == 200 else None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def search_books(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search Anna's Archive and return detailed book info including md5 hash.
        Enhanced with better parsing and metadata extraction.
        """
        logger.info(f"Searching Anna's Archive for: {query}")
        
        search_url = f"{self.base_url}/search"
        params = {
            'q': query,
            'sort': 'newest',  # Get newest/most relevant results
            'lang': 'en'  # Focus on English books
        }
        
        response = self._make_request(search_url, params)
        if not response:
            logger.error("Search request failed")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find book result containers (Anna's Archive uses specific div classes)
        book_containers = soup.find_all('div', class_=['mb-4', 'mb-3']) or soup.find_all('a', href=re.compile(r'/md5/[a-f0-9]{32}'))
        
        results = []
        seen_md5 = set()
        
        for container in book_containers:
            try:
                book_data = self._extract_book_info(container)
                if book_data and book_data['md5'] not in seen_md5:
                    seen_md5.add(book_data['md5'])
                    results.append(book_data)
                    
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.warning(f"Error parsing book container: {e}")
                continue

        logger.info(f"Found {len(results)} books for query: {query}")
        return results

    def _extract_book_info(self, container) -> Optional[Dict]:
        """Extract detailed book information from HTML container"""
        try:
            # Find MD5 hash from link
            md5_link = container.find('a', href=re.compile(r'/md5/[a-f0-9]{32}'))
            if not md5_link:
                return None
                
            md5_match = re.search(r'/md5/([a-f0-9]{32})', md5_link['href'])
            if not md5_match:
                return None
                
            md5 = md5_match.group(1)
            
            # Extract title (try multiple methods)
            title = "Unknown Title"
            title_element = (
                container.find('h3') or 
                container.find('div', class_='text-xl') or 
                md5_link
            )
            if title_element:
                title = title_element.get_text(strip=True)
                # Clean up title
                title = re.sub(r'\s+', ' ', title)
                title = title.replace('[', '').replace(']', '')
                if len(title) > 100:
                    title = title[:100] + "..."
            
            # Extract author
            author = "Unknown Author"
            author_patterns = [
                r'by\s+([^,\n\r]+)',
                r'Author[:\s]+([^,\n\r]+)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)'
            ]
            
            container_text = container.get_text()
            for pattern in author_patterns:
                author_match = re.search(pattern, container_text, re.IGNORECASE)
                if author_match:
                    author = author_match.group(1).strip()
                    break
            
            # Extract file format and size
            format_info = self._extract_format_info(container_text)
            
            # Extract year
            year = self._extract_year(container_text)
            
            # Extract language
            language = self._extract_language(container_text)
            
            return {
                "title": title,
                "author": author,
                "format": format_info.get('format', 'pdf'),
                "file_size": format_info.get('size', 'Unknown'),
                "md5": md5,
                "download_url": f"{self.base_url}/md5/{md5}",
                "year": year,
                "language": language,
                "quality": self._assess_quality(title, author, format_info)
            }
            
        except Exception as e:
            logger.error(f"Error extracting book info: {e}")
            return None

    def _extract_format_info(self, text: str) -> Dict[str, str]:
        """Extract file format and size from text"""
        format_match = re.search(r'\b(pdf|epub|mobi|djvu|txt|azw3)\b', text, re.IGNORECASE)
        format_type = format_match.group(1).lower() if format_match else 'pdf'
        
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*(MB|KB|GB)', text, re.IGNORECASE)
        size = f"{size_match.group(1)} {size_match.group(2).upper()}" if size_match else "Unknown"
        
        return {"format": format_type, "size": size}

    def _extract_year(self, text: str) -> str:
        """Extract publication year from text"""
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        return year_match.group(0) if year_match else "Unknown"

    def _extract_language(self, text: str) -> str:
        """Extract language from text"""
        language_patterns = [
            r'\b(English|Spanish|French|German|Italian|Russian|Chinese|Japanese)\b',
            r'\bLang[:\s]+([\w]+)',
            r'\bLanguage[:\s]+([\w]+)'
        ]
        
        for pattern in language_patterns:
            lang_match = re.search(pattern, text, re.IGNORECASE)
            if lang_match:
                return lang_match.group(1).capitalize()
        
        return "English"  # Default assumption

    def _assess_quality(self, title: str, author: str, format_info: Dict) -> str:
        """Assess book quality based on available information"""
        score = 0
        
        # Title quality
        if len(title) > 10 and title != "Unknown Title":
            score += 2
        
        # Author quality
        if author != "Unknown Author" and len(author.split()) >= 2:
            score += 2
        
        # Format preference
        format_scores = {'pdf': 2, 'epub': 3, 'mobi': 2, 'azw3': 2, 'djvu': 1, 'txt': 1}
        score += format_scores.get(format_info.get('format', 'pdf'), 1)
        
        # Size indicates quality (not too small, not too large)
        size = format_info.get('size', '')
        if 'MB' in size:
            try:
                mb = float(re.search(r'(\d+(?:\.\d+)?)', size).group(1))
                if 1 <= mb <= 50:  # Good size range
                    score += 1
            except:
                pass
        
        if score >= 6:
            return "High"
        elif score >= 4:
            return "Medium"
        else:
            return "Low"

    def get_download_links(self, md5_hash: str) -> Dict:
        """
        Get comprehensive download links for a specific book.
        Enhanced with multiple mirror detection and fallback options.
        """
        logger.info(f"Getting download links for MD5: {md5_hash}")
        
        book_url = f"{self.base_url}/md5/{md5_hash}"
        response = self._make_request(book_url)
        
        if not response:
            logger.error(f"Failed to fetch book page for MD5: {md5_hash}")
            return self._get_fallback_links(md5_hash)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        mirrors = []
        
        # Find download links on the page
        download_links = soup.find_all('a', href=True)
        
        for link in download_links:
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()
            
            # Identify different types of download links
            if any(keyword in text for keyword in ['download', 'get', 'mirror', 'link']):
                if href.startswith('http'):
                    mirrors.append({
                        "name": self._get_mirror_name(href, text),
                        "url": href,
                        "status": "active",
                        "type": self._get_link_type(href, text)
                    })
                elif href.startswith('/'):
                    mirrors.append({
                        "name": "Anna's Archive Direct",
                        "url": f"{self.base_url}{href}",
                        "status": "active",
                        "type": "direct"
                    })
        
        # Always include the main page as a fallback
        mirrors.insert(0, {
            "name": "Anna's Archive Main",
            "url": book_url,
            "status": "active",
            "type": "main"
        })
        
        # Remove duplicates
        unique_mirrors = []
        seen_urls = set()
        for mirror in mirrors:
            if mirror['url'] not in seen_urls:
                seen_urls.add(mirror['url'])
                unique_mirrors.append(mirror)
        
        return {
            "md5": md5_hash,
            "mirrors": unique_mirrors,
            "direct_download": f"{self.base_url}/download/{md5_hash}",
            "download_count": len(unique_mirrors)
        }

    def _get_mirror_name(self, url: str, text: str) -> str:
        """Determine mirror name from URL and text"""
        if 'libgen' in url.lower():
            return "Library Genesis"
        elif 'sci-hub' in url.lower():
            return "Sci-Hub"
        elif 'annas-archive' in url.lower():
            return "Anna's Archive"
        elif 'z-lib' in url.lower():
            return "Z-Library"
        elif text and len(text) > 3:
            return text.title()
        else:
            return "External Mirror"

    def _get_link_type(self, url: str, text: str) -> str:
        """Determine the type of download link"""
        if 'download' in url.lower() or 'download' in text:
            return "direct"
        elif 'mirror' in text or 'alternative' in text:
            return "mirror"
        else:
            return "redirect"

    def _get_fallback_links(self, md5_hash: str) -> Dict:
        """Provide fallback download options when main search fails"""
        return {
            "md5": md5_hash,
            "mirrors": [
                {
                    "name": "Anna's Archive",
                    "url": f"https://annas-archive.org/md5/{md5_hash}",
                    "status": "active",
                    "type": "main"
                },
                {
                    "name": "Library Genesis",
                    "url": f"http://libgen.rs/book/index.php?md5={md5_hash}",
                    "status": "fallback",
                    "type": "mirror"
                }
            ],
            "direct_download": f"https://annas-archive.org/download/{md5_hash}",
            "download_count": 2
        }

    def get_book_details(self, md5_hash: str) -> Optional[Dict]:
        """
        Get detailed information about a specific book.
        New method for enhanced book metadata.
        """
        logger.info(f"Getting detailed info for MD5: {md5_hash}")
        
        book_url = f"{self.base_url}/md5/{md5_hash}"
        response = self._make_request(book_url)
        
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        try:
            # Extract detailed metadata from the book page
            details = {
                "md5": md5_hash,
                "title": self._extract_detail_field(soup, ['title', 'Title']),
                "author": self._extract_detail_field(soup, ['author', 'Author', 'authors']),
                "publisher": self._extract_detail_field(soup, ['publisher', 'Publisher']),
                "year": self._extract_detail_field(soup, ['year', 'Year', 'published']),
                "pages": self._extract_detail_field(soup, ['pages', 'Pages']),
                "language": self._extract_detail_field(soup, ['language', 'Language', 'lang']),
                "isbn": self._extract_detail_field(soup, ['isbn', 'ISBN']),
                "format": self._extract_detail_field(soup, ['format', 'Format', 'extension']),
                "file_size": self._extract_detail_field(soup, ['size', 'Size', 'filesize']),
                "description": self._extract_description(soup)
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error extracting book details: {e}")
            return None

    def _extract_detail_field(self, soup: BeautifulSoup, field_names: List[str]) -> str:
        """Extract a specific field from book details page"""
        for field_name in field_names:
            # Try different HTML patterns
            patterns = [
                soup.find('span', string=re.compile(field_name, re.IGNORECASE)),
                soup.find('td', string=re.compile(field_name, re.IGNORECASE)),
                soup.find('div', string=re.compile(field_name, re.IGNORECASE))
            ]
            
            for pattern in patterns:
                if pattern:
                    # Get the next sibling or parent's next sibling
                    value_element = pattern.find_next_sibling() or pattern.parent.find_next_sibling()
                    if value_element:
                        return value_element.get_text(strip=True)
        
        return "Unknown"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract book description/summary"""
        desc_selectors = [
            'div[class*="description"]',
            'div[class*="summary"]',
            'p[class*="description"]',
            'div[id*="description"]'
        ]
        
        for selector in desc_selectors:
            desc_element = soup.select_one(selector)
            if desc_element:
                desc = desc_element.get_text(strip=True)
                if len(desc) > 50:  # Ensure it's substantial
                    return desc[:500] + "..." if len(desc) > 500 else desc
        
        return "No description available"

# Maintain backward compatibility
anna_archive = AnnaArchiveAPI()