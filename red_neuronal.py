import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.text as text

# --- Configuración de Matplotlib para modo interactivo ---
try:
    plt.ion()
    print("Modo interactivo de Matplotlib activado.")
except Exception as e:
    print(f"No se pudo activar el modo interactivo de Matplotlib: {e}")
    print("La visualización puede no actualizarse de forma fluida.")

# --- Mapeo de números a índices para one-hot encoding ---
# Usaremos 0 para el 3, 1 para el 4, 2 para el 5, 3 para el 6
input_value_to_index = {3: 0, 4: 1, 5: 2, 6: 3}
output_index_to_character = {0: "Guerrero", 1: "Mago"} # Definimos 0:Guerrero, 1:Mago para la salida softmax

# 1. Definir el modelo de red neuronal con 4 entradas y 2 salidas
# Esta estructura es apropiada para clasificar 4 entradas discretas en 2 categorías.
# NOTA IMPORTANTE: Este modelo NO está entrenado. Sus predicciones iniciales serán aleatorias.
# Para que aprenda la regla (>5 -> Mago, <=5 -> Guerrero), necesitaría ser entrenado
# con pares entrada-salida correspondientes (usando model.fit()).
model = keras.Sequential([
    # Capa de entrada: 4 neuronas, una por cada posible valor de entrada (3, 4, 5, 6)
    keras.layers.Input(shape=(len(input_value_to_index),)),
    # Capa oculta (opcional para este problema simple, pero incluida en el dibujo)
    keras.layers.Dense(4, activation='relu'), # Usamos 4 neuronas ocultas con activación ReLU
    # Capa de salida: 2 neuronas, una para Guerrero y una para Mago
    keras.layers.Dense(len(output_index_to_character), activation='softmax') # Softmax para salida de probabilidad
])

# 2. Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Usamos crossentropy para clasificación multiclase con softmax
              metrics=['accuracy'])

# 3. NO ESTABLECEMOS PESOS MANUALMENTE AQUÍ.
# Los pesos iniciales son aleatorios. La red NECESITA ser entrenada para aprender la lógica.
# Si la usas ahora, dará predicciones basadas en esos pesos aleatorios.

# 4. Función para convertir entrada numérica a one-hot encoding
def to_one_hot(nivel_ataque):
    one_hot = np.zeros(len(input_value_to_index))
    if nivel_ataque in input_value_to_index:
        one_hot[input_value_to_index[nivel_ataque]] = 1
    return one_hot.reshape(1, -1) # Asegura la forma correcta para el modelo (batch_size, num_features)

# 5. Usar la red neuronal para hacer predicciones (usando one-hot input)
def predecir_con_modelo(nivel_ataque_int):
     """
     Convierte el nivel de ataque a one-hot, lo pasa al modelo y retorna
     las probabilidades de salida y el personaje decidido.
     """
     if nivel_ataque_int not in input_value_to_index:
         return None, None, None # Indica entrada fuera del rango 3-6

     one_hot_input = to_one_hot(nivel_ataque_int)
     predictions = model.predict(one_hot_input, verbose=0)[0] # Obtener el array de probabilidades [prob_guerrero, prob_mago]

     # Decidir el personaje basado en la probabilidad más alta
     predicted_index = np.argmax(predictions)
     personaje = output_index_to_character[predicted_index]

     return personaje, predictions, predicted_index

# --- Función para dibujar la red neuronal con estructura 4-Input, Hidden, 2-Output ---
def draw_neural_network_structured(num_hidden_nodes=4):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Representación Estructura NN (4 Entradas, Capa Oculta, 2 Salidas) - No Entrenada")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-2, 2) # Más espacio vertical
    ax.axis('off')

    # Posiciones X para las capas
    layer_x = [0, 1.5, 3]

    # Posiciones Y para las neuronas
    num_inputs = len(input_value_to_index)
    num_outputs = len(output_index_to_character)
    input_y = np.linspace(-1, 1, num_inputs) # 4 nodos de entrada
    hidden_y = np.linspace(-0.8, 0.8, num_hidden_nodes) # Nodos ocultos
    output_y = np.linspace(-0.5, 0.5, num_outputs) # 2 nodos de salida

    node_positions = {
        'input': [(layer_x[0], y) for y in input_y],
        'hidden': [(layer_x[1], y) for y in hidden_y],
        'output': [(layer_x[2], y) for y in output_y]
    }

    node_radius = 0.12 # Radio ligeramente más pequeño para más nodos
    default_line_color = 'gray'
    default_line_width = 0.8

    # Diccionarios y listas para almacenar los objetos gráficos
    node_circles = {'input': [], 'hidden': [], 'output': []}
    connection_lines = [] # Guardamos todas las líneas

    # Dibujar neuronas y etiquetas de capa
    # Capa de Entrada (4 nodos)
    for i, (x, y) in enumerate(node_positions['input']):
        circle = patches.Circle((x, y), radius=node_radius, fc='lightblue', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        node_circles['input'].append(circle)
        # Etiqueta para el valor de entrada
        input_value = list(input_value_to_index.keys())[i]
        ax.text(x - 0.2, y, str(input_value), va='center', ha='right', fontsize=10, weight='bold')

    ax.text(layer_x[0], max(input_y) + 0.4, "Capa Entrada\n(Valor One-Hot)", ha='center', fontsize=11, weight='bold')

    # Capa Oculta (Conceptual)
    for (x, y) in node_positions['hidden']:
        circle = patches.Circle((x, y), radius=node_radius, fc='lightgreen', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        node_circles['hidden'].append(circle)
    ax.text(layer_x[1], max(hidden_y) + 0.4, f"Capa Oculta\n({num_hidden_nodes} Nodos)", ha='center', fontsize=11, weight='bold')

    # Capa de Salida (2 nodos)
    for i, (x, y) in enumerate(node_positions['output']):
        color = 'blue' if i == 0 else 'green' # Azul para Guerrero (índice 0), Verde para Mago (índice 1)
        circle = patches.Circle((x, y), radius=node_radius, fc=color, ec='black', linewidth=1.5)
        ax.add_patch(circle)
        node_circles['output'].append(circle)
        # Etiqueta para el personaje
        character_name = output_index_to_character[i]
        ax.text(x + 0.2, y, character_name, va='center', ha='left', fontsize=10, weight='bold')

    ax.text(layer_x[2], max(output_y) + 0.4, "Capa Salida\n(Softmax)", ha='center', fontsize=11, weight='bold')

    # Dibujar conexiones (todas las líneas)
    layers = ['input', 'hidden', 'output']
    for i in range(len(layers) - 1):
        from_layer = layers[i]
        to_layer = layers[i+1]
        for from_pos in node_positions[from_layer]:
            for to_pos in node_positions[to_layer]:
                 line = lines.Line2D([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                                     color=default_line_color, lw=default_line_width, zorder=-1, alpha=0.7) # Semi-transparente por defecto
                 ax.add_line(line)
                 connection_lines.append(line)


    # Añadir marcadores de texto para mostrar la entrada y la salida actuales
    input_text_label = ax.text(layer_x[0], -1.8, "Input: ?", ha='center', fontsize=12, color='blue', weight='bold')
    output_text_label = ax.text(layer_x[2], -1.8, "Output: ?", ha='center', fontsize=12, color='red', weight='bold')

    plt.show() # Mostrar la figura inicial

    # Devolvemos todos los objetos gráficos para poder actualizarlos
    return fig, ax, node_circles, connection_lines, input_text_label, output_text_label

# --- Función para actualizar la visualización (Resaltado limitado) ---
def update_network_visualization_structured(fig, ax, node_circles, connection_lines,
                                             input_text_label, output_text_label,
                                             input_value_int, predicted_character, predictions):
    """Actualiza los colores de nodos (entrada activada y salida ganadora) y textos."""

    # Definir colores de resaltado
    highlight_color_mago = 'green' # Color para la salida Mago
    highlight_color_guerrero = 'blue' # Color para la salida Guerrero
    default_node_color_input = 'lightblue'
    default_node_color_hidden = 'lightgreen'
    default_node_color_output_guerrero = 'blue' # Color base del nodo Guerrero
    default_node_color_output_mago = 'green' # Color base del nodo Mago
    default_line_color = 'gray'
    default_line_width = 0.8


    # --- Restablecer colores a sus valores por defecto ---
    # Restablecer nodos
    for circle in node_circles['input']:
        circle.set_facecolor(default_node_color_input)
        circle.set_edgecolor('black')
        circle.set_linewidth(1.5) # Ancho del borde por defecto

    for circle in node_circles['hidden']:
         circle.set_facecolor(default_node_color_hidden)
         circle.set_edgecolor('black')
         circle.set_linewidth(1.5)

    # Restablecer nodos de salida (usando sus colores base)
    if node_circles['output']:
        node_circles['output'][0].set_facecolor(default_node_color_output_guerrero)
        if len(node_circles['output']) > 1:
            node_circles['output'][1].set_facecolor(default_node_color_output_mago)

        for circle in node_circles['output']:
            circle.set_edgecolor('black')
            circle.set_linewidth(1.5)


    # Restablecer todas las líneas
    for line in connection_lines:
         line.set_color(default_line_color)
         line.set_linewidth(default_line_width)
         line.set_alpha(0.7)


    # --- Aplicar resaltado ---
    if input_value_int in input_value_to_index:
        # Resaltar el nodo de entrada correspondiente al valor
        input_node_index = input_value_to_index[input_value_int]
        if input_node_index < len(node_circles['input']):
            node_circles['input'][input_node_index].set_facecolor('yellow') # Resaltar entrada
            node_circles['input'][input_node_index].set_edgecolor('red')
            node_circles['input'][input_node_index].set_linewidth(2.5)


    if predicted_character:
        # Resaltar el nodo de salida que "ganó"
        if predicted_character == "Guerrero" and len(node_circles['output']) > 0:
            node_circles['output'][0].set_facecolor('gold') # Resaltar nodo Guerrero
            node_circles['output'][0].set_edgecolor('red')
            node_circles['output'][0].set_linewidth(2.5)
        elif predicted_character == "Mago" and len(node_circles['output']) > 1:
            node_circles['output'][1].set_facecolor('gold') # Resaltar nodo Mago
            node_circles['output'][1].set_edgecolor('red')
            node_circles['output'][1].set_linewidth(2.5)

        # NOTA: No intentamos resaltar todas las conexiones o nodos ocultos porque es muy complejo
        # mapear la decisión de la red no entrenada a un camino visual significativo aquí.


    # Actualizar texto en el gráfico
    input_text_label.set_text(f"Input: {input_value_int}")
    output_text_label.set_text(f"Output: {predicted_character}\nProb: {predictions}") # Mostrar probabilidades completas

    # Intentar forzar una actualización del lienzo
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# --- Parte de interacción manual ---
print("--- Red Neuronal Estructurada (No Entrenada) ---")
print("Introduce un nivel de ataque (entre 3 y 6).")
print("Escribe 'salir' para terminar.")

# Dibujar la red con la nueva estructura
fig, ax, node_circles, connection_lines, input_text_label, output_text_label = draw_neural_network_structured(num_hidden_nodes=4)

while True:
    entrada = input("Introduce el nivel de ataque: ")

    if entrada.lower() == 'salir':
        plt.close(fig) # Cerrar la figura de matplotlib al salir
        break

    try:
        nivel = int(entrada)

        # Validar el rango de entrada (debe ser 3, 4, 5, o 6)
        if nivel not in input_value_to_index:
            print(f"Entrada fuera de rango: {nivel}. Por favor, introduce un número entero entre 3 y 6.")
            continue

        # Usar la función de predicción con el modelo (NO ENTRENADO)
        personaje_predicho, predictions, predicted_index = predecir_con_modelo(nivel)

        # Mostrar el resultado de la predicción del modelo actual (aleatorio) en consola
        print(f"Nivel de ataque introducido: {nivel}")
        print(f" -> Probabilidades de salida del modelo (Guerrero, Mago): {predictions}")
        print(f" -> Personaje predicho por el modelo (aleatorio sin entrenar): {personaje_predicho}")


        # --- Actualizar la visualización ---
        if personaje_predicho is not None: # Si la entrada estaba en el rango válido
             update_network_visualization_structured(fig, ax, node_circles, connection_lines,
                                                      input_text_label, output_text_label,
                                                      nivel, personaje_predicho, predictions)


    except ValueError:
        print("Entrada no válida. Por favor, introduce un número entero entre 3 y 6 o 'salir'.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


print("Programa terminado.")
