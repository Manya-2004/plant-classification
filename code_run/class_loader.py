import os


DATASET_PATH = "data"


def load_classes():

    dataset = {}

    categories = os.listdir(DATASET_PATH)

    for category in categories:

        category_path = os.path.join(
            DATASET_PATH,
            category
        )

        if not os.path.isdir(category_path):
            continue

        plants = []

        for plant in os.listdir(category_path):

            plant_path = os.path.join(
                category_path,
                plant
            )

            if os.path.isdir(plant_path):

                plants.append(plant)

        dataset[category] = sorted(plants)

    return dataset


if __name__ == "__main__":

    classes = load_classes()

    print(classes)