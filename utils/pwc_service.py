import os
import time
import json
import requests

class PapersWithCodeClient:
    BASE_URL = "https://paperswithcode.com/api/v1"

    def __init__(self, task_slug: str, data_dir: str = "data"):
        self.task_slug = task_slug
        self.data_dir = data_dir
        self.papers_url = f"{self.BASE_URL}/tasks/{task_slug}/papers/"
        self.pdf_folder = os.path.join(data_dir, "papers_pdfs")
        os.makedirs(self.pdf_folder, exist_ok=True)

    def fetch_json(self, url: str):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def fetch_papers_metadata(self, limit: int | None) -> list[dict]:
        url = self.papers_url
        papers_list = []
        
        while url:
            response = self.fetch_json(url)
            if not response:
                break

            results = response["results"]
            for paper in results:
                entry = {
                    "Title": paper["title"],
                    "Authors": ", ".join(paper["authors"]),
                    "Abstract": paper.get("abstract", "No abstract available"),
                    "PDF URL": paper["url_pdf"],
                    "Datasets": [],
                    "Tasks": [],
                    "Tasks id": [],
                }

                paper_slug = paper["id"]

                # Fetch datasets
                dataset_url = f"{self.BASE_URL}/papers/{paper_slug}/datasets/"
                dataset_data = self.fetch_json(dataset_url)
                if dataset_data:
                    entry["Datasets"] = [d["name"] for d in dataset_data["results"]]

                # Fetch tasks
                task_url = f"{self.BASE_URL}/papers/{paper_slug}/tasks/"
                # print(task_url)
                task_data = self.fetch_json(task_url)
                # print(task_data)
                if task_data:
                    entry["Tasks"] = [t["name"] for t in task_data["results"]]
                    entry["Tasks id"] = [t["id"] for t in task_data["results"]]

                papers_list.append(entry)

                if limit is not None and len(papers_list) >= limit:
                    return papers_list[:limit]

                time.sleep(1)  # avoid rate-limiting

            url = response.get("next")

        return papers_list

    def save_json(self, data: list[dict], filename: str):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Saved data to {filepath}")

    def load_json(self, filename: str) -> list[dict]:
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def download_pdf(self, pdf_url: str, title: str) -> str | None:
        safe_title = title.replace(" ", "_").replace("/", "_")
        filename = os.path.join(self.pdf_folder, f"{safe_title}.pdf")

        try:
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
            return filename
        except requests.RequestException as e:
            print(f"Failed to download {title}: {e}")
            return None

    def download_all_pdfs(self, papers_list: list[dict]) -> list[dict]:
        for paper in papers_list:
            pdf_url = paper.get("PDF URL")
            title = paper["Title"]

            if pdf_url and pdf_url.startswith("http"):
                pdf_path = self.download_pdf(pdf_url, title)
                paper["Local PDF Path"] = pdf_path if pdf_path else None
            else:
                print(f"No valid PDF URL for: {title}")
                paper["Local PDF Path"] = None
        return papers_list
