"""
Wrong Side Driving Detection using YOLOv4
Module to handle firebase operations for our project
Done By,
Sriram N C
Srinandan KS
Jyoti Shetty
R.V. College of Engineering, Bangalore
"""

from firebase import firebase
import pyrebase
import json2table
import json
import os
import webbrowser


class FirebaseKeyNotFoundException(Exception):
    """
    Custom exception class to handle Firebase-side errors, usually related to authentication
    """
    pass


def retrieve_data_json(db, db_id) -> str:
    """
    Retrieve JSON data from Firebase
    :param db: Database URL
    :param db_id: Database ID
    :return: Retrieved JSON data
    """
    json_str = ""
    cars = db.child(db_id).get()
    for car in cars.each():
        json_str += str(car.val())
        json_str += ','
    json_str = json_str.replace("'", "\"")
    json_str = json_str[:-1]

    return json_str


def display_json_data_on_webpage(json_str: str) -> int:
    """
    Display json data retrieved in a web browser
    :param json_str: json data as a string
    :return: 0 for success
    """
    json_data = json.loads(json_str)
    build_direction = "LEFT_TO_RIGHT"
    table_attributes = {"style": "width:100%"}
    html = json2table.convert(json_data, build_direction=build_direction, table_attributes=table_attributes)
    path = os.path.abspath('temp.html')
    url = 'file://' + path
    with open(path, 'w') as f:
        f.write(html)
    webbrowser.get("/usr/bin/google-chrome").open(url)
    return 0


def add_record(storage, firebaseb, car_id, timestamp, my_image, db_id, user, table="Violations") -> str:
    """
    Add a record to the Firebase DB
    :param storage: Firebase storage object
    :param firebaseb: Firebase authentication object
    :param car_id: ID of car added to DB (after LPR which will be added in the near future)
    :param timestamp: Timestamp of violation commission
    :param my_image: Image of violation as proof
    :param db_id: DB specific unique ID
    :param user: User object for Firebase DB's authenticated user
    :param table: Name of DB table in which record should be stored
    :return: URL where image is stored in DB
    """
    d = os.getcwd()
    os.chdir(os.path.dirname(my_image))
    my_image = os.path.basename(my_image)
    storage.child(my_image).put(my_image)
    url = storage.child(my_image).get_url(user['idToken'])
    # print(url)
    data = {
        "ID": car_id,
        "timestamp": timestamp,
        "image": url
    }
    firebaseb.post('/{}/{}'.format(db_id, table), data)
    os.chdir(d)
    print("DB record added successfully")
    return url


def delete_object(record_id, db, db_id, table="Violations"):
    """
    Delete DB record
    :param record_id: ID of the record to be deleted
    :param db: Firebase DB object
    :param db_id: DB specific unique ID
    :param table: Name of DB table in which record should be deleted
    :return: None
    """
    database_records = db.child(db_id).child(table).get()
    my_key = None
    for data in database_records.each():
        if data.val()['ID'] == record_id:
            my_key = data.key()

    if my_key is None:
        raise FirebaseKeyNotFoundException("Given id does not exist in the database records")

    db.child(db_id).child(table).child(my_key).remove()
    print("Deleted record successfully")


def authenticate() -> tuple:
    """
    Authenticate Firebase user using sensitive_data.json
    :return: tuple consisting of database object to be used, db id, 2 objects created during authentication, Firebase Storage object and user object
    """
    with open(os.path.join(os.getcwd(), 'sensitive_data.json')) as f:  # Keep the json file in the same parent dir of this code
        json_dict = json.load(f)

    firebase_config = json_dict["firebaseConfig"]
    db_id = json_dict["db_id"]

    # initializing
    firebasea = pyrebase.initialize_app(firebase_config)
    db = firebasea.database()
    storage = firebasea.storage()
    auth = firebasea.auth()
    email = json_dict["email"]
    password = json_dict["password"]
    user = auth.sign_in_with_email_and_password(email, password)
    firebaseb = firebase.FirebaseApplication(firebase_config["databaseURL"], None)
    return db, db_id, firebasea, firebaseb, storage, user


def main():
    print("Functions here are: 1.Retrieve database details on a webpage")
    print("2.Retrieve database details in JSON\n3.Update the database with a new Violation")
    print("4. Delete row by id")
    choice = int(input())
    # config files
    db, db_id, firebasea, firebaseb, storage, user = authenticate()

    if choice == 1:
        # webpage display
        print("Retrieve data as webpage\n")
        json_data_str = retrieve_data_json(db, db_id)
        display_json_data_on_webpage(json_data_str)

    if choice == 2:
        # json result
        print("Retrieve data as JSON\n")
        retrieve_data_json(db, db_id)

    if choice == 3:
        # adding a violation
        print("Adding in a violation")
        car_id = input("Enter car ID: ")
        timestamp = input("Enter timestamp of detection: ")
        my_image = input("Enter the path of image to be added in : ")
        add_record(storage, firebaseb, car_id, timestamp, my_image, db_id, user)

    if choice == 4:
        # deleting a violation
        print("Deleting a violation")
        vehicle_id = input("Enter the ID of the vehicle to be removed: ")
        delete_object(vehicle_id, db, db_id)


if __name__ == '__main__':
    main()
