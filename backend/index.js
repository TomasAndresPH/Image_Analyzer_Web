const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Configuración de multer para manejar las cargas de imágenes
const upload = multer({ dest: 'uploads/' });

// Variable global para el modelo cargado
let model;

// Clases del modelo
const classes = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges'];

// Función para cargar el modelo
const loadModel = async () => {
  try {
    console.log('Cargando modelo...');
    model = await tf.loadLayersModel('file://modelo_js/model.json');
    console.log('Modelo cargado exitosamente.');
  } catch (error) {
    console.error('Error al cargar el modelo:', error);
  }
};

// Función para preprocesar imágenes
const preprocessImage = (imageBuffer) => {
  const imageTensor = tf.node.decodeImage(imageBuffer)
    .resizeBilinear([28, 28]) // Cambiar dimensiones según el modelo
    .toFloat()
    .div(tf.scalar(255.0)) // Normalizar entre 0 y 1
    .expandDims(0); // Añadir dimensión de batch
  return imageTensor;
};

// Endpoint para cargar una imagen y obtener predicción
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    const filePath = req.file.path; // Ruta temporal del archivo cargado
    const imageBuffer = fs.readFileSync(filePath);

    // Preprocesar la imagen
    const imageTensor = preprocessImage(imageBuffer);

    // Realizar predicción
    const predictions = model.predict(imageTensor).dataSync();
    const predictedIndex = predictions.indexOf(Math.max(...predictions));
    const predictedLabel = classes[predictedIndex];
    const probability = predictions[predictedIndex];

    // Limpiar archivo temporal
    fs.unlinkSync(filePath);

    // Enviar respuesta
    res.json({ 
      prediction: predictedLabel, 
      probability: probability.toFixed(2) 
    });
  } catch (error) {
    console.error('Error al procesar la imagen:', error);
    res.status(500).json({ error: 'Error al realizar la predicción.' });
  }
});

// Iniciar el servidor
app.listen(PORT, async () => {
  await loadModel();
  console.log(`Servidor corriendo en http://localhost:${PORT}`);
});