import json
import ollama
import asyncio
from vnstock3 import Vnstock
from datetime import date

# Simulates an API call to get flight times
# In a real application, this would fetch data from a live database or API
def get_flight_times(departure: str, arrival: str) -> str:
  flights = {
    'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
    'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
    'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
    'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
    'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
    'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
  }

  key = f'{departure}-{arrival}'.upper()
  return json.dumps(flights.get(key, {'error': 'Flight not found'}))
def get_antonyms(word: str) -> str:
    "Get the antonyms of the any given word"

    words = {
        "hot": "cold",
        "small": "big",
        "weak": "strong",
        "light": "dark",
        "lighten": "darken",
        "dark": "bright",
    }

    return json.dumps(words.get(word, "Not available in database"))

def get_stock_price(stock, price_date=None):
    # Khoi tao stock object
    stock_obj = Vnstock().stock(symbol=stock.upper(), source="VCI")
    if price_date is not None:
        get_date = price_date
    else:
        get_date = date.today().strftime('%Y-%m-%d')

    df = stock_obj.quote.history(start = get_date, end=get_date, interval = "1D")
    response_json = {"stock": stock, "price": str(df['close'][0]*1000) + " VND", "last_update": get_date}
    # print(response_json)

    return json.dumps(response_json)

async def run(model: str):
  client = ollama.AsyncClient()
  # Initialize conversation with a user query
  messages = [{'role': 'user', 'content': 'What is the sky is blue'}]

  # First API call: Send the query and function description to the model
  response = await client.chat(
    model=model,
    messages=messages,
    tools=[
      {
        'type': 'function',
        'function': {
          'name': 'get_flight_times',
          'description': 'Get the flight times between two cities',
          'parameters': {
            'type': 'object',
            'properties': {
              'departure': {
                'type': 'string',
                'description': 'The departure city (airport code)',
              },
              'arrival': {
                'type': 'string',
                'description': 'The arrival city (airport code)',
              },
            },
            'required': ['departure', 'arrival'],
          },
        },
      },
            {
                "type": "function",
                "function": {
                    "name": "get_antonyms",
                    "description": "Get the antonyms of any given words",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {
                                "type": "string",
                                "description": "The word for which the opposite is required.",
                            },
                        },
                        "required": ["word"],
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get the current price in a given stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock": {
                                "type": "string",
                                "description": "The stock, e.g. SHB, SHS",
                            },
                            "price_date": {
                                "type": "string",
                                "description": "The date to get price of stock, e.g. 2024-08-11",
                            },
                        },
                        "required": ["stock"],
                    },
                },
            }
    ],
  )

  # Add the model's response to the conversation history
  messages.append(response['message'])

  # Check if the model decided to use the provided function
  if not response['message'].get('tool_calls'):
    print("The model didn't use the function. Its response was:")
    print(response['message']['content'])
    return

  # Process function calls made by the model
  if response['message'].get('tool_calls'):
    available_functions = {
        'get_flight_times': get_flight_times,
        "get_antonyms": get_antonyms,
        "get_stock_price": get_stock_price,
    }
    for tool in response['message']['tool_calls']:
        function_to_call = available_functions[tool['function']['name']]
        if function_to_call == get_flight_times:
            function_response = function_to_call(
                tool["function"]["arguments"]["departure"],
                tool["function"]["arguments"]["arrival"],
            )
            print(f"function response: {function_response}")

        elif function_to_call == get_antonyms:
            function_response = function_to_call(
                tool["function"]["arguments"]["word"],
            )
            print(f"function response: {function_response}")
        elif function_to_call == get_stock_price:
            function_response = function_to_call(
                tool["function"]["arguments"]["stock"],
                tool["function"]["arguments"]["price_date"],
            )
            print(f"function response: {function_response}")
      # Add function response to the conversation
    messages.append(
    {
        'role': 'tool',
        'content': function_response,
    }
    )

  # Second API call: Get final response from the model
  final_response = await client.chat(model=model, messages=messages)
  print("second response\n")
  print(final_response['message']['content'])


# Run the async function
asyncio.run(run('llama3.1'))