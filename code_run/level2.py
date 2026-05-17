import os
import shutil

# -------------------------
# PATHS
# -------------------------
SRC = r"_data_\plant_metadata\Healthy"     # your original dataset (90 classes)
DST = r"data"

# -------------------------
# NORMALIZATION
# -------------------------
def normalize(name):
    return name.lower().replace(" ", "_").strip()

# -------------------------
# SAME MAPPING (reuse)
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
# BUILD SOURCE CLASS MAP
# -------------------------
print("\n🔍 Scanning source dataset...")
src_classes = {
    normalize(folder): folder
    for folder in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, folder))
}

print(f"Found {len(src_classes)} classes\n")

# -------------------------
# CREATE LEVEL-2 DATASETS
# -------------------------
missing_classes = []
total_images = 0

for category, plants in mapping.items():
    category_path = os.path.join(DST, category)
    os.makedirs(category_path, exist_ok=True)

    print(f"\n📂 Creating dataset: {category}")

    for plant in plants:
        norm_plant = normalize(plant)

        if norm_plant not in src_classes:
            print(f"  ❌ Missing: {plant}")
            missing_classes.append(plant)
            continue

        real_folder = src_classes[norm_plant]
        src_path = os.path.join(SRC, real_folder)

        dst_plant_path = os.path.join(category_path, norm_plant)
        os.makedirs(dst_plant_path, exist_ok=True)

        images = os.listdir(src_path)

        for img in images:
            src_img = os.path.join(src_path, img)
            dst_img = os.path.join(dst_plant_path, img)

            shutil.copy2(src_img, dst_img)
            total_images += 1

        print(f"  ✅ {plant} → {len(images)} images")

# -------------------------
# SUMMARY
# -------------------------
print("\n==============================")
print("✅ LEVEL-2 DATASETS CREATED")
print(f"Total images copied: {total_images}")

if missing_classes:
    print("\n⚠️ Missing classes:")
    for m in missing_classes:
        print(f" - {m}")
else:
    print("\n🎉 All classes processed successfully!")