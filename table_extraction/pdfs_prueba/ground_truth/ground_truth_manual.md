# Guía de Anotación de Tablas

## 1. Objetivo

El objetivo de este proceso es construir un ground truth estructurado en formato JSON para evaluar sistemas de extracción de tablas (por ejemplo, DeepDoctection, Camelot, LLMs, entre otros). Las anotaciones deben ser consistentes, reproducibles y completamente alineadas con el esquema definido.

---

## 2. Estructura general del JSON

Cada anotación debe seguir estrictamente la siguiente estructura:

{
  "documents": [
    {
      "paper_title": "...",
      "num_tables": N,
      "tables": [
        {
          "table_id": "...",
          "page": X,
          "evaluation": {
            "expected_rows": N,
            "expected_cols": M,
            "columns": [...]
          },
          "rows": [...]
        }
      ]
    }
  ]
}

No se deben agregar campos adicionales fuera de esta estructura.

---

## 3. Nivel documento

### paper_title

- Debe coincidir exactamente con el título original del paper
- No se permite traducir, abreviar o modificar el texto

### num_tables

- Número total de tablas presentes en el documento
- Debe coincidir exactamente con la cantidad de elementos en "tables"

---

## 4. Nivel tabla

Cada tabla debe incluir los siguientes campos obligatorios:

- table_id
- page
- evaluation
- rows

### table_id

- Formato obligatorio: table_1, table_2, table_3, ...
- La numeración reinicia en cada documento
- Debe seguir el orden de aparición en el PDF

### page

- Número de página donde aparece la tabla
- Debe ser un número entero
- Basado en la numeración visible del PDF

---

## 5. Bloque evaluation

Define la estructura esperada de la tabla:

"evaluation": {
  "expected_rows": N,
  "expected_cols": M,
  "columns": [...]
}

### expected_rows

- Número de filas de datos
- No incluye encabezados
- Debe coincidir con len(rows)

### expected_cols

- Número de columnas

### columns

Lista de nombres de columnas.

Reglas:

- Deben seguir el orden visual de izquierda a derecha

---

## 6. Aplanamiento de columnas

Si la tabla contiene encabezados multinivel, deben combinarse en un solo nombre de columna.

Incorrecto:

["Mean", "Mean", "Hits@10"]

Correcto:

["WN18_Mean_raw", "WN18_Mean_filter", "WN18_Hits@10_raw", "WN18_Hits@10_filter"]

Convención recomendada:

Dataset_Metrica_Variante

Ejemplo:

WN18_Hits@10_filter

---

## 7. Filas (rows)

Formato:

"rows": [
  ["valor1", valor2, valor3, ...]
]

### Reglas generales

- Cada fila debe ser una lista
- Cada fila debe tener exactamente expected_cols elementos
- El orden debe coincidir con el de las columnas

---

## 8. Tipos de datos

### Texto

- Debe mantenerse exactamente como aparece en el paper
- Incluir referencias como [6], (Ji et al., 2015), etc.

Ejemplo:

"RESCAL Nickel et al. [22]"

---

### Valores numéricos

- Deben almacenarse como números, no como texto

Correcto:

75.4

Incorrecto:

"75.4"

---

### Valores faltantes

Se debe usar exactamente:

"-"

No usar:

- null
- NaN
- valores vacíos

Ejemplo:

["ConvE*", "-", "-", "-", 95.6, "-", "-", "-", "-"]

---

## 9. Casos especiales

### Columnas sin nombre

Si una columna no tiene encabezado, se debe usar un string vacío:

"columns": ["", "WN18", "FB15k"]

---

### Símbolos especiales

Se deben preservar exactamente como aparecen:

- ξ
- σ
- λ
- γ
- β

---

### Expresiones

Las expresiones deben mantenerse como texto:

"[n/2]"

---

### Unidades

Las unidades deben mantenerse exactamente:

"0.4GB"

---

## 10. Flujo de anotación

Paso 1: Identificar todas las tablas del documento

Paso 2: Asignar table_id en orden de aparición

Paso 3: Extraer las columnas y aplicar flatten si es necesario

Paso 4: Extraer todas las filas

Paso 5: Completar:

- expected_rows
- expected_cols

---

## 11. Reglas de consistencia

### Regla 1: No inferir

- No completar valores faltantes
- No corregir errores del paper
- No interpretar datos

---

### Regla 2: Literalidad

- Copiar exactamente lo que aparece
- Respetar mayúsculas, símbolos y formato

---

### Regla 3: Consistencia global

- Usar las mismas convenciones en todo el dataset
- Especialmente para:
  - valores faltantes ("-")
  - nombres de columnas

---

## 12. Control de calidad

Antes de finalizar cada tabla:

- Verificar expected_rows
- Verificar expected_cols
- Verificar que todas las filas tengan la misma longitud
- Verificar que el orden de columnas coincida con los datos

---

## 13. Exclusiones

No incluir:

- captions de tablas
- títulos
- notas al pie
- texto fuera de la tabla

---

## 14. Consideraciones finales

La anotación debe seguir estrictamente este formato. El JSON resultante debe ser directamente utilizable para evaluación automática sin necesidad de transformaciones adicionales.