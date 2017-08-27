import shelve

DB_PATH = 'data/db'

def find_all_keys():
    """Find all the documents in the shelve database."""
    with shelve.open(DB_PATH) as db:
        for key in list(db.keys()):
            print(db[key])
    return None

def read_key(key):
    """Find all the documents in the shelve database."""
    with shelve.open(DB_PATH) as db:
        try:
            return db[key]
        except KeyError:
            return False

def create_key(key, value):
    """Add a new key-value pair to the shelve database."""
    with shelve.open(DB_PATH) as db:
        db[key] = value
    return True

def update_key(key, value):
    """Update the value for a key in the shelve database."""
    with shelve.open(DB_PATH) as db:
        db[key] = value
    return True

def delete_key(key):
    """Delete the item associated with a key in a Mongo database."""
    with shelve.open(DB_PATH) as db:
        del db[key]
    return True

def delete_all_keys():
    """Find all the documents in the shelve database."""
    with shelve.open(DB_PATH) as db:
        for key in list(db.keys()):
            del db[key]
    return True
