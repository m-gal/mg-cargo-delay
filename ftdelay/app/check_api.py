"""
    Check the API with FastAPI

    1. Run ./app/main.py
    2. Run from Terminal in ./ftdelay:
            >>> python app/check_api.py
        or in ./ftdelay/app:
            >>> python check_api.py

    Open http://127.0.0.1:8000/docs.

    or Insert example values into 'Request body'
    * For one-value example a mResponse body must be:
    {
        "target": {
            "proba_delay": ["proba"],
            "dpd_delay": [
                "[00]",
                "[01-03]",
                "[04-07]",
                "[08-14]",
                "[15-21]",
                "[22+]",
            ],
        },
        "prediction": [
            {
                "proba_delay": [0.4787444472312927],
                "dpd_delay": [
                    0.45605671405792236,
                    0.5428952574729919,
                    0.0001441613130737096,
                    0.00021221564384177327,
                    0.0005598959396593273,
                    0.00013174983905628324,
                ],
            }
        ],
    }
    @author: mikhail.galkin
"""
# * One-value example values testing:
instances = [
    {
        "weight_kg": [850.0],
        "ade_month": ["FEB"],
        "carrier_code": ["UASI"],
        "container_id_prefix": ["MATU"],
        "container_type_of_service": ["CS"],
        "place_of_receipt": ["SHANGHAI,CN"],
        "port_of_lading": ["SHANGHAI"],
        "port_of_unlading": ["LONG BEACH CA"],
        "vessel_name": ["MANULANI"]
    }
]

# * Two-values example values testing:
# instances = [
#     {
#         "weight_kg": [850.0],
#         "ade_month": ["FEB"],
#         "carrier_code": ["UASI"],
#         "container_id_prefix": ["MATU"],
#         "container_type_of_service": ["CS"],
#         "place_of_receipt": ["SHANGHAI,CN"],
#         "port_of_lading": ["SHANGHAI"],
#         "port_of_unlading": ["LONG BEACH CA"],
#         "vessel_name": ["MANULANI"]
#     },
#     {
#         "weight_kg": [13255.1],
#         "ade_month": ["JAN"],
#         "carrier_code": ["ZIMU"],
#         "container_id_prefix": ["ZCSU"],
#         "container_type_of_service": ["PP"],
#         "place_of_receipt": ["NINGBO (ZJ)"],
#         "port_of_lading": ["NINGPO"],
#         "port_of_unlading": ["SAVANNAH GA"],
#         "vessel_name": ["TIANJIN"]
#     }
# ]

if __name__ == "__main__":
    import requests

    response = requests.post(
        "http://127.0.0.1:8000/us_delay/predict",
        json=instances,
    )
    print(response.content)

"""
Interactive API docs:
Now to get the above result we had to manually call each endpoint but FastAPI
comes with Interactive API docs which can access by adding /docs in your path.
To access docs for our API we'll go to http://127.0.0.1:8000/docs.
Here you'll get the following page where you can test the endpoints of your API
by seeing the output they'll give for the corresponding inputs if any.
"""
