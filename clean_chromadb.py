import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


settings=Settings(is_persistent=True)
client = chromadb.Client(settings=settings,tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,)
# Initialize the Chroma client. Make sure your Chroma DB server is running.


def get_all_collection_names():
    """Retrieves a list of all collection names in the Chroma DB."""
    collections = client.list_collections()
    return collections #[collection.name for collection in collections]

def delete_collection_by_name(collection_name):
    """Deletes a Chroma collection given its name.

    Args:
        collection_name: The name of the collection to delete.
    """
    try:
        client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except ValueError as e:
        print(f"Error deleting collection '{collection_name}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 1. Get and print all collection names
    all_collections = get_all_collection_names()
    if all_collections:
        print("Existing Collections:")
        for collection_name in all_collections:
            print(f"- {collection_name}")
    else:
        print("No collections found.")


    # 2. Prompt user to enter a collection name to delete.  Handle empty input.
    collection_to_delete = input("Enter the name of the collection to delete (or press Enter to skip): ").strip()

    if collection_to_delete:
      # Basic validation: Prevent deleting collections with obviously wrong names
      if len(collection_to_delete) < 2 :
         print("Collection names should have at least two characters")

      elif collection_to_delete not in get_all_collection_names():
          print(f"Collection '{collection_to_delete}' does not exist and cannot be deleted.")
      else:
          # 3. Delete the specified collection
          delete_collection_by_name(collection_to_delete)

          #  Show updated collection list
          all_collections = get_all_collection_names()
          if all_collections:
              print("Existing Collections after deletion:")
              for collection_name in all_collections:
                  print(f"- {collection_name}")
          else:
              print("No collections found.")
    else:
        print("No collection name entered. Nothing deleted.")
