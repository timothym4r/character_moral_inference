import json
import pandas as pd
from rapidfuzz import process, fuzz

def construct_structured_data(new_dialogue, old_dialogue, moral_rating_data, subjects, similarity_threshold):
    """
    Constructs a structured DataFrame from new and old dialogue data along with moral ratings.
    
    params:
    - new_dialogue: dict, new dialogue data with character names as keys and sentences as values.
    - old_dialogue: dict, old dialogue data with character names as keys and sentences as values.
    - moral_rating_data: pd.DataFrame, DataFrame containing moral ratings with character codes and ratings.
    - similarity_threshold: int, threshold for string similarity to match character names.
    - subjects: dict, dictionary containing subject codes and names for movies along with their characters.
    
    returns:
    - dict, structured data with movie names as keys and movie and character details as values.

    """
    structured_data = {}
    count = 0
    count2 = 0
    count3 = 0

    movies_in_subjects = []  # List to store movie names in moral rating data

    pairs = []
    for subject, subject_data in subject.items():
        movies_in_subjects.append((subject, subject_data["name"]))

    # Go through movies in moral rating data and see if they match with the new dialogue data or old dialogue data
    for movie_code, movie_name in movies_in_subjects:
        for movie_1 in new_dialogue.keys():
            parts = movie_1.rsplit("_", 1)
            movie = parts[0]
            year = parts[1]

            # Check if the movie name matches
            if movie_name.lower() == movie.lower():
                pairs.append((movie_1, (movie_code, movie_name)))
                break
        else:
            for movie_2 in old_dialogue.keys():
                movie = movie_2

                # Check if the movie name matches
                if movie_name.lower() == movie.lower():
                    pairs.append((movie_2, (movie_code, movie_name)))
                    break
            else:
                print(f"Movie {movie_name} not found in new or old dialogue data.")

    # Now we have pairs of (subject_code, subject_name) and movie name
    for pair in pairs:
        dct = {}
        dct["subject_name"] = pair[1][1]  # The corresponding title in the subject.json of the movie
        dct["subject_code"] = pair[1][0]  # The corresponding code in the subject.json of the movie
        dct["characters"] = {}

        # Extract the movie name from the pair
        # This will also work if there is no underscore in the movie name
        movie_name = pair[0].rsplit("_", 1)[0]

        if movie_name in structured_data:
            print(f"Movie {movie_name} already exists in structured_data. Skipping...")
            continue

        for char_index in subjects[pair[1][0]]:
            if char_index == "name" or char_index == "N":  # skip the name and n keys since they are not characters
                continue

            # Find the corresponding character name in moral_character_dict_100
            # We can use string similarity or find fraction of the character name
            character_subject_name = subjects[pair[1][0]][char_index]
            if isinstance(character_subject_name, list):  # Ensure it's a list before accessing the first element
                character_subject_name = character_subject_name[0]
            else:
                continue
            
            # Use substring matching to find the best match for the character name
            best_match = []
            
            found_match_in_new_dialogue = True
            # We should make sure the character here has at least 50 sentences in the new dialogue data
            for char_name, data in new_dialogue[pair[0]].items():
                if char_name.lower() in character_subject_name.lower():
                    best_match = char_name  # Mock similarity score to pass the threshold
                    break
                else:
                    score = fuzz.token_sort_ratio(char_name.lower(), character_subject_name.lower())
                    if score >= similarity_threshold:
                        best_match.append((char_name, score))
                        break
            else:
                found_match_in_new_dialogue = False
                # If no match found in new dialogue, check in old dialogue
                for char_name, data in old_dialogue[pair[0]].items():
                    if char_name.lower() in character_subject_name.lower():
                        best_match = char_name
                    else:
                        score = fuzz.token_sort_ratio(char_name.lower(), character_subject_name.lower())
                        if score >= similarity_threshold:
                            best_match.append((char_name, score))
                else:
                    print(f"No match found for character: {character_subject_name} in movie: {pair[1][1]}")
                    continue
            
            if best_match:
                print(f"Found match for character: {character_subject_name} in movie: {pair[1][1]} with best match: {best_match}")
                count+=1
            else:
                print(f"No match found for character: {character_subject_name} in movie: {pair[1][1]}")
                continue
            
            # The below code is not reached even if the best_match is list
            # TODO: fix this if we want to use the best match from the list
            if isinstance(best_match, list) and len(best_match) > 0:
                # Sort the matches by similarity score and take the best one
                best_match = sorted(best_match, key=lambda x: x[1], reverse=True)[0][0]  # Get the character name with the highest score
                count += 1  # This does not add up the count, meaning we found all the match not here
            elif isinstance(best_match, list) and len(best_match) == 0:
                print(f"No match found for character: {character_subject_name} in movie: {pair[1][1]}")
                continue

            char_name = best_match
            # Once we find the character, we store the character code, name in dialogue data, name in subjects, rating, and the latent embeddings
            char_dct = {}
            char_dct["character_name"] = char_name
            # Use the corresponding character name to find the embedding
            char_dct["subject_name"] = character_subject_name
            char_dct["subject_code"] = pair[1][0] + "/" + str(char_index)

            rating_data = moral_rating_data[moral_rating_data["character_code"] == char_dct["subject_code"]]
            rating_data = rating_data.iloc[0].tolist()
            char_dct["rating"] = rating_data[1:]  # Skip the character_code column
            
            if found_match_in_new_dialogue:
                char_dct["sentences"] = new_dialogue[pair[0]][char_name]
            else:
                char_dct["sentences"] = old_dialogue[pair[0]][char_name]

            # probably there are duplicates in the character names, so we overwrite the character data
            if char_name in dct["characters"]:
                print(f"Character {char_name} already exists in movie {movie_name}. Overwriting...")
            
            dct["characters"][char_name] = char_dct
            
        structured_data[movie_name] = dct        
    return structured_data
    
def add_movies_to_structured_data(structured_data, movies_to_add, new_dialogue, old_dialogue, moral_rating_data, subjects):
    """
    Add movies to the structured data from the new dialogue data.

    params:
    - structured_data: dict, existing structured data.
    - movies_to_add: list, list of movie names to add to the structured data. format: [movie_name_in_dialogue, movie_code_in_rating, old_or_new), ...]
    - new_dialogue: dict, new dialogue data with character names as keys and sentences as values.
    - old_dialogue: dict, old dialogue data with character names as keys and sentences as values.
    - moral_rating_data: pd.DataFrame, DataFrame containing moral ratings with character codes and ratings.
    - subjects: dict, dictionary containing subject codes and names for movies along with their characters.
    returns:
    - dict, updated structured data with new movies added.
    """
    for movie_info in movies_to_add:
        movie_name_in_dialogue = movie_info[0]
        movie_code_in_rating = movie_info[1]
        old_or_new = movie_info[2]

        if movie_name_in_dialogue not in new_dialogue and movie_name_in_dialogue not in old_dialogue:
            print(f"Movie {movie_name_in_dialogue} not found in new or old dialogue data.")
            continue

        if movie_code_in_rating not in subjects:
            print(f"Movie with code {movie_code_in_rating} not found in subjects data.")
            continue

        # Create a new entry for the movie
        dct = {}
        dct["subject_name"] = subjects[movie_code_in_rating]["name"]
        dct["subject_code"] = movie_code_in_rating
        dct["characters"] = {}

        # Add characters from the dialogue data
        if old_or_new == "new":
            dialogue_data = new_dialogue[movie_name_in_dialogue]
        else:
            dialogue_data = old_dialogue[movie_name_in_dialogue]

        for char_name, sentences in dialogue_data.items():
            char_dct = {}
            char_dct["character_name"] = char_name
            char_dct["subject_name"] = char_name  # Assuming character name is the same as subject name
            char_dct["subject_code"] = f"{movie_code_in_rating}/{char_name}"  # Assuming character code is movie_code/character_name
            
            rating_data = moral_rating_data[moral_rating_data["character_code"] == char_dct["subject_code"]]
            if not rating_data.empty:
                rating_data = rating_data.iloc[0].tolist()
                char_dct["rating"] = rating_data[1:]  # Skip the character_code column
            else:
                print(f"No rating data found for character {char_dct['subject_code']}.")

            char_dct["sentences"] = sentences
            
            dct["characters"][char_name] = char_dct
        
        structured_data[movie_name_in_dialogue] = dct
    
    return structured_data

if __name__ == "__main__":

    with open("../data/new_dialogue.json", "r") as f:
        new_dialogue = json.load(f)
    
    with open("../data/old_dialogue.json", "r") as f:
        old_dialogue = json.load(f)

    # Load the Moral Rating Data
    moral_rating_file_path = "../data/SWCPQ-Features-Survey-Dataset-November2023/data files/features-survey-dataset.csv"
    moral_rating_data = pd.read_csv(moral_rating_file_path)

    # Load the subjects data
    moral_rating_file_path = "../data/SWCPQ-Features-Survey-Dataset-November2023/data files/subjects.json"
    with open("../data/subjects.json", "r") as f:
        subjects = json.load(f)
    
    # Construct the structured data
    similarity_threshold = 80  # Set the similarity threshold for matching character names
    structured_data = construct_structured_data(new_dialogue, old_dialogue, moral_rating_data, subjects, similarity_threshold)
    print(f"Constructed structured data with {len(structured_data)} movies.")

    # Save the structured data to a JSON file
    structured_data_file_save_path = "../data/structured_data.json"

    # OPTIONAL: If you want to add more movies to the structured data, you can do so here
    # Example of movies to add: [("movie_name_in_dialogue", "movie_code_in_rating", "old_or_new"), ...]
    movies_to_add = [
        # Example: ("Inception_2010", "inception", "new"),
        # Add more movies as needed
        ]   
    
    structured_data = add_movies_to_structured_data(structured_data, movies_to_add, new_dialogue, old_dialogue, moral_rating_data, subjects)

    with open(structured_data_file_save_path, "w") as f:
        json.dump(structured_data, f, indent=2)
    
