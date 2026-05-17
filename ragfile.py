import json
import os
from collections import Counter
from difflib import get_close_matches


class PlantRAG:

    def __init__(self):

        # =====================================================
        # BASE PATH
        # =====================================================

        self.base_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "metadata"
                )
            )

        # =====================================================
        # JSON FILES
        # =====================================================

        self.disease_file = os.path.join(self.base_path, "disease.json")
        self.health_file = os.path.join(self.base_path, "health.json")
        self.plants_file = os.path.join(self.base_path, "healthy_plants.json")

        # =====================================================
        # LOAD JSON FILES
        # =====================================================

        with open(self.disease_file, "r", encoding="utf-8") as f:
            self.disease_data = json.load(f)

        with open(self.health_file, "r", encoding="utf-8") as f:
            self.health_data = json.load(f)

        with open(self.plants_file, "r", encoding="utf-8") as f:
            self.plants_data = json.load(f)

        # =====================================================
        # CLASS MAPPING
        # =====================================================

        self.class_mapping = {

            "bamboo": "Bamboo (बांस)",
            "betel": "Betel Leaf (पान)",
            "betel_nut": "Areca Nut (सुपारी)",
            "common_ruenaagdalli": "Spurge (नागदाली)",
            "coriender": "Coriander (धनिया)",
            "curry_leaf": "Curry Leaves (करी पत्ता)",
            "lemon_grass": "Lemongrass (लेमनग्रास)",
            "mint": "Mint (पुदीना)",
            "wood_sorel": "Wood Sorrel (खट्टी बूटी)",

            "geranium": "Geranium (जेरैनियम)",
            "globe_amarnath": "Globe Amaranth (गुलदाउदी अमरनाथ)",
            "hibiscus": "Hibiscus (गुड़हल)",
            "jasmine": "Jasmine (चमेली)",
            "marigold": "Marigold (गेंदा)",
            "nithyapushpa": "Periwinkle (सदाबहार)",
            "parijatha": "Night-flowering Jasmine (रात की रानी)",
            "rose": "Rose (गुलाब)",
            "sampige": "Jasmine (चमेली)",
            "tecoma": "Tecoma (पीला कनेर)",
            "thumbe": "Globe Amaranth (गुलदाउदी अमरनाथ)",

            "aloe_vera": "Aloe Vera (एलोवेरा)",
            "amruthaballi": "Giloy (गिलोय)",
            "arali": "Oleander (कनेर)",
            "ashwagandha": "Ashwagandha (अश्वगंधा)",
            "badipala": "Milk Hedge (सीमा बाड़ पौधा)",
            "brahmi": "Brahmi (ब्राह्मी)",
            "bringaraja": "Bhringraj (भृंगराज)",
            "camphor": "Camphor Tree (कपूर)",
            "castor": "Castor Plant (अरंडी)",
            "catharanthus": "Periwinkle (सदाबहार)",
            "doddapatre": "Indian Borage (अजवाइन पत्ता)",
            "ginger": "Ginger (अदरक)",
            "henna": "Henna (मेहंदी)",
            "insulin": "Insulin Plant (इंसुलिन पौधा)",
            "kamakasturi": "Musk Mallow (कस्तूरी भिंडी)",
            "kasambruga": "Balloon Vine (कनफटा बेल)",
            "malabar_nut": "Malabar Nut (अडूसा)",
            "nagadali": "Spurge (नागदाली)",
            "nelavembu": "Kalmegh (कालमेघ)",
            "tulsi": "Tulsi",
            "turmeric": "Turmeric (हल्दी)",

            "ashoka": "Ashoka Tree (अशोक)",
            "coffee": "Coffee Plant (कॉफी पौधा)",
            "eucalyptus": "Eucalyptus (नीलगिरी)",
            "guava": "Guava (अमरूद)",
            "honge": "Indian Beech (करंज)",
            "indian_gooseberry": "Indian Gooseberry (आंवला)",
            "jackfruit": "Jackfruit (कटहल)",
            "lemon": "Lemon (नींबू)",
            "mango": "Mango (आम)",
            "neem": "Neem (नीम)",
            "nerale": "Jamun (जामुन)",
            "papaya": "Papaya (पपीता)",
            "pomegranate": "Pomegranate (अनार)",
            "sapota": "Sapodilla (चीकू)",
            "seethaashoka": "Ashoka Tree (सीता अशोक)",
            "tamarind": "Tamarind (इमली)",

            "basella": "Malabar Spinach (बासले)",
            "beans": "Beans (सेम)",
            "chilly": "Chili Pepper (मिर्च)",
            "drumstick": "Moringa (ड्रमस्टिक)",
            "gasagase": "Poppy Seeds (खसखस)",
            "kohlrabi": "Kohlrabi (गांठ गोभी)",
            "malabar_spinach": "Malabar Spinach (बासले)",
            "onion": "Onion (प्याज)",
            "pea": "Pea (मटर)",
            "pepper": "Black Pepper (काली मिर्च)",
            "pumpkin": "Pumpkin (कद्दू)",
            "raddish": "Radish (मूली)",
            "spinach": "Spinach (पालक)",
            "taro": "Taro (अरबी)"
        }

    # =========================================================
    # GET DISEASE INFO
    # =========================================================

    def get_disease_info(self, disease_name):

        diseases = self.disease_data.get("plant_diseases", {})

        match = get_close_matches(
            disease_name,
            diseases.keys(),
            n=1,
            cutoff=0.4
        )

        if not match:
            return "Disease not found"

        disease_key = match[0]

        data = diseases[disease_key]

        result = f"""
Disease Name: {disease_key}

Plant: {data.get("plant", "N/A")}
Scientific Cause: {data.get("scientific_cause", "N/A")}

Symptoms:
"""

        for symptom in data.get("symptoms", []):
            result += f"  - {symptom}\n"

        result += f"\nCauses: {data.get('causes', 'N/A')}\n"

        result += "\nPrevention:\n"

        for item in data.get("prevention", []):
            result += f"  - {item}\n"

        result += "\nTreatment:\n"

        for item in data.get("treatment", []):
            result += f"  - {item}\n"

        result += f"\nCategory: {data.get('category', 'N/A')}\n"

        return result

    # =========================================================
    # GET PLANT INFO
    # =========================================================

    def get_plant_info(self, plant_name):

        match = get_close_matches(
            plant_name.lower(),
            self.plants_data.keys(),
            n=1,
            cutoff=0.4
        )

        if not match:
            return "Plant not found"

        plant_key = match[0]

        plant = self.plants_data[plant_key]

        # mapped name
        display_name = self.class_mapping.get(
            plant_key,
            plant_key.replace("_", " ").title()
        )

        result = f"""
{display_name}

Scientific Name: {plant.get("scientific_name", "N/A")}

Medicinal Uses:
{plant.get("medicinal_uses", "N/A")}

Culinary Uses:
{plant.get("Culinary", "N/A")}

Ornamental Uses:
{plant.get("Ornamental", "N/A")}

Health Benefits:
{plant.get("Health benefits", "N/A")}

Nutritional Value:
{plant.get("Nutritional value", "N/A")}

Description:
{plant.get("description", "N/A")}

Category:
{plant.get("category", "N/A")}

Care Instructions:
{plant.get("care_instructions", "N/A")}

Disease Prevention:
{plant.get("disease_prevention", "N/A")}

Usage:
{plant.get("usage", "N/A")}


Disease Prevention:
"""

        prevention = plant.get("disease_prevention", [])

        if isinstance(prevention, list):
            for item in prevention:
                result += f"  - {item}\n"
        else:
            result += f"  - {prevention}\n"

        result += f"""

Usage:
{plant.get("usage", "N/A")}
"""

        return result

    # =========================================================
    # RECOMMEND PLANTS
    # =========================================================

    def recommend_plants(self, issue_input, top_k=5):

        issue_input = issue_input.lower()

        matched_issues = []

        for issue in self.health_data.keys():

            if issue in issue_input or issue_input in issue:
                matched_issues.append(issue)

        if not matched_issues:
            return "No matching issues found"

        all_plants = []

        for issue in matched_issues:
            all_plants.extend(self.health_data[issue])

        ranked = Counter(all_plants)

        top_plants = ranked.most_common(top_k)

        result = f"\nMatched Issues: {', '.join(matched_issues)}\n\n"

        result += "Recommended Plants:\n\n"

        for idx, (plant_name, score) in enumerate(top_plants, start=1):

            display_name = self.class_mapping.get(
                plant_name,
                plant_name.replace("_", " ").title()
            )

            result += f"{idx}. {display_name}  (Score: {score})\n"

        return result