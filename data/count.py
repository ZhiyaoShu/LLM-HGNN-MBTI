import json

# Open the JSON file
with open('users_data_all.json') as file:
    data = json.load(file)

# Count the number of dictionaries/items
count = len(data)

# Print the count
print(f"The number of dictionaries/items in users_data_all.json is: {count}")