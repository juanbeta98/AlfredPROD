
# Data Dictionary

| Column Name       | Description                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------|
| service_id        | Identificador del servicio; un servicio puede contener distintos labor ids (de la tabla services)     |
| labor_id          | Identificador de cada labor                                                                          |
| labor_type        | Número entero para identificar cada tipo de labor                                                    |
| labor_name        | Nombre del labor                                                                                     |
| labor_category    | Categoría a la que pertenece el labor                                                                |
| labor_price       | Precio de realizar el labor                                                                          |
| labor_created_at  | Fecha en la que se creó el registro del labor                                                        |
| labor_start_date  | Fecha en la que inició el labor                                                                      |
| labor_end_date    | Fecha en la que finalizó el labor                                                                    |
| alfred            | Identificador del Alfred asignado al labor                                                           |
| shop              | Identificador del lugar donde se realizó el servicio                                                 |
| created_at        | Fecha en la que se creó el servicio                                                                  |
| schedule_date     | Fecha en la que se planeó realizar el servicio                                                       |
| client_type       | Tipo de cliente (B2B o B2C)                                                                          |
| paying_customer   | Identificador del cliente                                                                            |
| state_service     | Estado de servicio ('CANCELED', 'COMPLETED', 'CATTLED', 'TO_BE_PAID', 'TO_BE_CONFIRMED', 'IN_PROGRESS', 'REQUESTED') |
| start_address_id  | Identificador de la dirección inicial del servicio                                                   |
| start_address_point | Longitud y latitud del punto inicial del servicio                                                  |
| end_address_id    | Identificador de la dirección final del servicio                                                     |
| end_address_point | Longitud y latitud del punto final del servicio                                                      |
| city              | Identificador de la ciudad (149 para Bogotá)                                                         |
| service_mode      | Modo de servicio ('WITH_DRIVER', 'WITHOUT_DRIVER')                                                   |
| address_id        | Identificador de dirección del labor                                                                 |
| address_point     | Longitud y latitud del punto del labor                                                               |
| address_name      | Nombre del sitio del labor                                                                           |
| start_to_point    | Distancia en KM de start_address_point a address_point                                               |
| point_to_end      | Distancia en KM de address_point a end_address_point                                                 |
| start_to_end      | Distancia en KM de start_address_point a end_address_point                                           |
