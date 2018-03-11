import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from pymongo import MongoClient
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


def bayes(userList,test_size):
    accuracy = 0

    for user in userList:

        review = np.array(user[1:])
        X = review[:, 1:25]
        Y = review[:, 0]
        Y = np.array(Y)

        clf = MultinomialNB()

        cv = ShuffleSplit(n_splits=10, test_size=test_size, random_state=40)

        scores = cross_val_score(clf, X, Y, cv=cv)

        accuracy = accuracy + scores.mean()
        
    print("Naive Bayes:")
    print(accuracy / len(userList) * 100)



def decision_tree(userList,test_size,depth,leaf):
    accuracy = 0

    for user in userList:
        user_id = user[0]
        review = np.array(user[1:])
        X = review[:, 1:25]
        Y = review[:, 0]
        Y = np.array(Y)

        clf = DecisionTreeClassifier(criterion="gini",
                                          max_depth=depth, min_samples_leaf=leaf)

        cv = ShuffleSplit(n_splits=10, test_size=test_size, random_state=40)

        scores = cross_val_score(clf, X, Y, cv=cv)

        accuracy = accuracy + scores.mean()

    print("Desicion Tree:")
    print(accuracy / len(userList) * 100)



def neural(userList,test_size,max,min):
    accuracy = 0

    for user in userList:

        review = np.array(user[1:])
        X = review[:, 1:25]
        Y = review[:, 0]
        Y = np.array(Y)

        clf = MLPClassifier(solver='sgd', alpha=1e-5,learning_rate = "constant",
                            learning_rate_init=0.005,
                            hidden_layer_sizes=(max,min), random_state=40)

        cv = ShuffleSplit(n_splits=2, test_size=test_size, random_state=0)

        scores = cross_val_score(clf, X, Y, cv=cv)

        accuracy = accuracy + scores.mean()

    print("Neural Network:")
    print(str(accuracy / len(userList) * 100))



def data(db,gte,lt):
    user_list = []

    cursor = db.toronto.find({"count": {"$gte": gte, "$lt": lt}})
    for document in cursor:

        user = []
        id = document["user_id"]
        user.append(id)

        for i in document["business"]:
            review_list = []
            ii = "" + i
            business = document["business"][ii]

            if business['stars'] == 1 or business['stars'] == 2:
                review_list.append(0)
            elif business['stars'] == 3 or business['stars'] == 4:
                review_list.append(1)
            else:
                review_list.append(2)

            # review_list.append(business['stars'])
            bus_attr = business['attribute']
            goodMeal = bus_attr['GoodForMeal']
            bus_park = bus_attr['BusinessParking']

            review_list.append(bus_attr['Alcohol'])
            review_list.append(bus_attr['HasTV'])
            review_list.append(bus_attr['NoiseLevel'])
            review_list.append(bus_attr['Caters'])
            review_list.append(bus_attr['WiFi'])
            review_list.append(bus_attr['RestaurantsReservations'])
            review_list.append(bus_attr['BikeParking'])
            review_list.append(bus_attr['RestaurantsTakeOut'])
            review_list.append(bus_attr['GoodForKids'])
            review_list.append(bus_attr['RestaurantsTableService'])
            review_list.append(bus_attr['OutdoorSeating'])
            review_list.append(bus_attr['RestaurantsPriceRange2'])
            review_list.append(bus_attr['RestaurantsDelivery'])
            review_list.append(bus_attr['WheelchairAccessible'])

            review_list.append(goodMeal['dessert'])
            review_list.append(goodMeal['latenight'])
            review_list.append(goodMeal['breakfast'])
            review_list.append(goodMeal['dinner'])
            review_list.append(goodMeal['lunch'])
            review_list.append(goodMeal['brunch'])

            review_list.append(bus_park['garage'])
            review_list.append(bus_park['street'])
            review_list.append(bus_park['validated'])
            review_list.append(bus_park['lot'])
            review_list.append(bus_park['valet'])

            user.append(review_list)
        user_list.append(user)


    neural(user_list, 0.3, 25, 100)

    bayes(user_list,0.3)

    decision_tree(user_list,0.3,2,5)

    print("-------------")


if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017/')
    db = client['yelp']

    data(db, 40, 70)
    data(db, 70, 100)
    data(db, 100, 200)
    data(db, 200, 300)
    data(db, 300, 1000)

