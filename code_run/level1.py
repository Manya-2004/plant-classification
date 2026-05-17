import os
import shutil
import uuid

# -------------------------
# PATHS (CHANGE IF NEEDED)
# -------------------------
SRC = r"_data_\plant_metadata\Healthy"
DST = r"data\level1"

# -------------------------
# NORMALIZATION FUNCTION
# -------------------------
def normalize(name):
    return name.lower().replace(" ", "_").strip()

# -------------------------
# CATEGORY MAPPING
# -------------------------
mapping = {
    "trees_fruits": [
        "mango", "guava", "jackfruit", "sapota", "pomegranate",
        "papaya", "avocado", "tamarind", "neem", "ashoka",
        "seethaashoka", "indian_gooseberry", "nerale", "honge",
        "eucalyptus", "coffee", "lemon", "citron_lime"
    ],

    "herbs_medicinal": [
        "tulsi", "aloe_vera", "ashwagandha", "brahmi", "bringaraja",
        "amruthaballi", "insulin", "henna", "turmeric", "ginger",
        "malabar_nut", "nelavembu", "kamakasturi", "nagadali",
        "kasambruga", "doddapatre", "arali", "badipala", "camphor",
        "castor", "catharanthus"
    ],

    "vegetables": [
        "beans", "onion", "pumpkin", "raddish", "pea",
        "chilly", "spinach", "basella", "malabar_spinach",
        "kohlrabi", "taro", "drumstick", "pepper", "gasagase"
    ],

    "flowers": [
        "rose", "jasmine", "hibiscus", "marigold", "geranium",
        "nithyapushpa", "parijatha", "sampige", "tecoma",
        "globe_amarnath", "thumbe"
    ],

    "weeds": [
        "lantana", "astma_weed", "ekka", "balloon_vine",
        "caricature", "chakte", "kepala", "ganigale",
        "ganike", "raktachandini", "padri", "kambajala"
    ],

    "climbers": [
        "betel", "betel_nut", "curry_leaf", "lemon_grass",
        "mint", "coriender", "wood_sorel",
        "common_ruenaagdalli", "bamboo"
    ]
}

# -------------------------
# SCAN SOURCE DATASET
# -------------------------
print("\n🔍 Scanning source dataset...")

src_classes = {
    normalize(folder): folder
    for folder in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, folder))
}

print(f"✅ Found {len(src_classes)} classes\n")

# -------------------------
# CREATE DESTINATION
# -------------------------
os.makedirs(DST, exist_ok=True)

missing_classes = []
copied_count = 0

# -------------------------
# BUILD LEVEL-1 DATASET
# -------------------------
for category, plants in mapping.items():
    category_path = os.path.join(DST, category)
    os.makedirs(category_path, exist_ok=True)

    print(f"\n📂 Processing category: {category}")

    for plant in plants:
        norm_plant = normalize(plant)

        if norm_plant not in src_classes:
            print(f"  ❌ Missing: {plant}")
            missing_classes.append(plant)
            continue

        real_folder = src_classes[norm_plant]
        plant_path = os.path.join(SRC, real_folder)

        images = os.listdir(plant_path)

        success = 0

        for img in images:
            src_img = os.path.join(plant_path, img)

            # Skip non-files
            if not os.path.isfile(src_img):
                continue

            # Get extension
            ext = os.path.splitext(img)[1]

            # Create short unique filename (fixes Windows error)
            new_name = f"{norm_plant}_{uuid.uuid4().hex[:8]}{ext}"
            dst_img = os.path.join(category_path, new_name)

            try:
                shutil.copy2(src_img, dst_img)
                copied_count += 1
                success += 1
            except Exception as e:
                print(f"    ⚠️ Error copying {img}: {e}")

        print(f"  ✅ {plant} → {success} images")

# -------------------------
# SUMMARY
# -------------------------
print("\n==============================")
print("✅ LEVEL-1 DATASET CREATED")
print(f"📊 Total images copied: {copied_count}")

if missing_classes:
    print("\n⚠️ Missing classes:")
    for m in missing_classes:
        print(f" - {m}")
else:
    print("\n🎉 All classes mapped successfully!")