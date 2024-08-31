import json
import base64
import io
import requests
import boto3
import uuid 
import logging
import hashlib
from dotenv import load_dotenv

# Configuración de logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.client('dynamodb')
s3_client = boto3.client('s3')
 
import os
# Cargar las variables de entorno desde el archivo .env
load_dotenv()
# Acceder a las variables de entorno
OPENIA_KEY = os.getenv('OPENIA_KEY')

def lambda_handler(event, context):
    image_data = base64.b64decode(event['Image'])
    logger.info(f"Event received: {event}")
    logger.info(f"Received image with size: {len(image_data)} bytes")
    bucket_name = "identificacionimagenes3"
    received_checksum = event['Checksum']
    producto = event['Product']
           
    # Calculate checksum
    calculated_checksum = hashlib.md5(image_data).hexdigest()
         
    try: 
        # Verificar el tamaño de la imagen
        if len(image_data) > 20 * 1024 * 1024:  # 20 MB
            raise ValueError("La imagen supera el tamaño permitido de 20 MB")
        
        # Valid extensions for OpenAI
        valid_extensions = ['png', 'jpeg', 'gif', 'webp']
        
        # Get the extension of the image from the event
        extensionFile = event.get('Extension', 'jpeg').lower()
        
        logger.info(f"Image extension: {extensionFile}")
         
        key = f"upload/{uuid.uuid4()}.{extensionFile}"
        s3_client.put_object(Body=image_data, Bucket=bucket_name, Key=key)
        
        image_url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
        
        logger.info(f"Image uploaded to S3: {image_url}")
        
        # Verify checksum
        if received_checksum != calculated_checksum:
            raise ValueError(f"el checksum no coincide. llegada:{received_checksum} - actual:{calculated_checksum}")
        
        if extensionFile not in valid_extensions:
            raise ValueError(f"Unsupported image format. Supported formats are: png, jpeg, gif, webp: -{extensionFile}-")
        
        productoId = obtener_id_producto(producto)
        
        #hacer consulta a dynamo por la llave 
        response = dynamodb.query(
            TableName='imagenesDeteccionTable',
            KeyConditionExpression='Id = :Id',
            ExpressionAttributeValues={
                ':Id': {'N': str(productoId)}
            }
        )
        
        filtered_items = response['Items']        
        
        if not filtered_items:
            raise ValueError(f"No se encontraron elementos para el producto: {producto}")
        
        prompt = filtered_items[0]['prompt']['S']
        model = filtered_items[0]['modelo']['S']
        baseImage = filtered_items[0]['baseImage']['S']
        
        url_producto_base = f"https://{bucket_name}.s3.amazonaws.com/upload/{baseImage}.png"
        
        #remplazar el texto {0} con lo que tiene la variable producto en mayusculas en el prompt
        prompt = prompt.replace("{0}", producto.upper())
        
        resultado = generar_texto_con_gpt4(image_url, prompt, model, url_producto_base, producto)
        return {
            'statusCode': 200,
            'body': json.dumps(resultado)
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"una excepcion lanzada": str(e)})
        }

def obtener_id_producto(producto):
    #hacer un switch para obtener el id del producto
    switcher = {
        "vive100": 1,
        "cocacola": 2,
        "aguila": 3,
        "magi": 4,
    }
    
    return switcher.get(producto.lower(), 0)

def generar_texto_con_gpt4(image_url, prompt, model, baseImage, producto):
         
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENIA_KEY}"
    } 

    if producto.lower() == "cualquiera":
        messages_content = [
            {
                "type": "text",
                "text": f"{prompt}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    else:
        messages_content = [
            {
                "type": "text",
                "text": f"{prompt}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": baseImage
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]

    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": messages_content
            }
        ],
        "max_tokens": 4000
    }
    
    logger.info(f"Request payload: {payload}")
    
    try:
        logger.info(f"Sending request to OpenAI API")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        valjson = response.json()
        
        logger.info(f"Response from OpenAI API: {valjson}")
        
        if "choices" in valjson and len(valjson["choices"]) > 0:
            response = valjson["choices"][0]["message"]["content"]
            logger.info(f"Response from OpenAI API: {response}")
            return response
        else:
            logger.warning(f"La respuesta de la API no contiene la estructura esperada: {valjson}")
            return f"La respuesta de la API no contiene la estructura esperada. {valjson}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la solicitud a la API: {e}")
        error_response = None
        
        # Intentar capturar más detalles del error
        if e.response is not None:
            try:
                error_response = e.response.json()
                logger.error(f"Detalle del error en JSON: {error_response}")
            except ValueError:
                error_response = e.response.text
                logger.error(f"Detalle del error en texto: {error_response}")
        else:
            logger.error("No se recibió una respuesta válida de la API.")
        
        return {
            "error": "Error en la solicitud a la API",
            "details": error_response
        }
