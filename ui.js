import * as tf from '@tensorflow/tfjs'

export const drawCanvas = (data) => {
    const canvas = document.getElementById('canvas')
    const ctx = canvas.getContext('2d')
    const tileWidth = data.shape[1]
    const tileWCount = data.shape[0]
    for (let i = 0; i < tileWCount; i++) {
        const image = data.slice([i, 0], [1, data.shape[1]])
        draw(image.flatten(), canvas)
    }
}

export const draw = (image, canvas) => {
    return tf.tidy(() => {
        const [width, height] = [28, 28]
    
        const ctx = canvas.getContext('2d')
        const imageData = new ImageData(width, height)
        const data = image//.dataSync()
        for (let i = 0; i < height * width; ++i) {
            const j = i * 4
            imageData.data[j + 0] = data[i] * 255
            imageData.data[j + 1] = data[i] * 255
            imageData.data[j + 2] = data[i] * 255
            imageData.data[j + 3] = 255
        }
        // console.log(imageData)
        ctx.putImageData(imageData, 0, 0)
    })
}

export const showTestResults = (data) => {
    if(!data) return
    const samples = data.shape[0]
    const imgs = []
    for (let i = 0; i < samples; i++) {        
        const image = data.slice([i, 0], [1, data.shape[1]])
        const canvas = document.createElement('canvas')
        canvas.width = 28
        draw(image.flatten(), canvas)
        imgs.push(canvas)
    }
    return imgs
}

export const getImages = (data) => {
    return tf.tidy(() => {
        const samples = data.shape[0]
        const imgs = []
        for (let i = 0; i < samples; i++) {        
            const image = data.slice([i, 0], [1, data.shape[1]])
            imgs.push(image.flatten().dataSync())
        }
        // console.log(imgs)
        return imgs
    })
}

// export const showTestResults = (batch, predictions, labels) => {
//     const testExamples = batch.xs.shape[0];
//     imagesElement.innerHTML = '';
//     for (let i = 0; i < testExamples; i++) {
//       const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);
  
//       const div = document.createElement('div');
//       div.className = 'pred-container';
  
//       const canvas = document.createElement('canvas');
//       canvas.className = 'prediction-canvas';
//       draw(image.flatten(), canvas);
  
//       const pred = document.createElement('div');
  
//       const prediction = predictions[i];
//       const label = labels[i];
//       const correct = prediction === label;
  
//       pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
//       pred.innerText = `pred: ${prediction}`;
  
//       div.appendChild(pred);
//       div.appendChild(canvas);
  
//       imagesElement.appendChild(div);
//     }
//   }