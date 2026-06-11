import requests
import os
import time

# --- Config ---
MAIN_FAMILIES = {
    "Carabidae":     49567,
    "Chrysomelidae": 51146,
    "Staphylinidae": 47951,
}

OTHER_FAMILY_NAMES = [
    "Aderidae", "Anthicidae", "Cantharidae", "Cerambycidae",
    "Cleridae", "Coccinellidae", "Colydiidae", "Curculionidae",
    "Eucinetidae", "Histeridae", "Lagriidae", "Monommidae",
    "Nitidulidae", "Pedilidae", "Phalacridae", "Platypodidae",
    "Ptilodactylidae", "Scarabaeidae", "Scolytidae", "Tenebrionidae",
]

MAIN_IMAGES_PER_FAMILY = 100
OTHER_IMAGES_PER_FAMILY = 15
OUTPUT_DIR = "beetle_images"
API = "https://api.inaturalist.org/v2/observations"
TAXA_API = "https://api.inaturalist.org/v2/taxa"


def lookup_taxon_id(name):
    """Look up a taxon ID by name from iNaturalist (v1 API)."""
    params = {"q": name, "rank": "family", "per_page": 1}
    r = requests.get("https://api.inaturalist.org/v1/taxa", params=params)
    r.raise_for_status()
    results = r.json().get("results", [])
    if results:
        taxon = results[0]
        print(f"  Found: {taxon['name']} (ID: {taxon['id']})")
        return taxon["id"]
    else:
        print(f"  WARNING: Could not find taxon ID for '{name}', skipping.")
        return None

def download_image(url, filepath):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def scrape_family(label, taxon_id, target_count, folder):
    print(f"\nScraping {label} (taxon {taxon_id})...")
    os.makedirs(folder, exist_ok=True)

    collected = 0
    page = 1

    while collected < target_count:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "per_page": 50,
            "page": page,
            "fields": "id,photos.url",
        }
        response = requests.get(API, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            print(f"  No more results at page {page}.")
            break

        for obs in results:
            if collected >= target_count:
                break
            photos = obs.get("photos", [])
            if not photos:
                continue
            url = photos[0]["url"].replace("square", "large")
            ext = url.split(".")[-1].split("?")[0]
            filename = os.path.join(folder, f"{label}_{obs['id']}.{ext}")
            if download_image(url, filename):
                collected += 1
                print(f"  [{collected}/{target_count}] Saved {filename}")
            time.sleep(0.3)

        page += 1

    print(f"  Done. {collected} images saved for {label}.")


# --- Main families ---
for label, taxon_id in MAIN_FAMILIES.items():
    folder = os.path.join(OUTPUT_DIR, label)
    scrape_family(label, taxon_id, MAIN_IMAGES_PER_FAMILY, folder)
    

# --- Other families: look up IDs then scrape ---
print("\n--- Looking up 'other' family taxon IDs ---")
other_families = {}
for name in OTHER_FAMILY_NAMES:
    taxon_id = lookup_taxon_id(name)
    if taxon_id:
        other_families[name] = taxon_id
    time.sleep(0.3)

for label, taxon_id in other_families.items():
    folder = os.path.join(OUTPUT_DIR, label)
    scrape_family(label, taxon_id, OTHER_IMAGES_PER_FAMILY, folder)

print("\nAll done!")