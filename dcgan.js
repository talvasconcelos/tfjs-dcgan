import * as tf from '@tensorflow/tfjs'

tf.setBackend('cpu')

class DCGAN {
    constructor(opts) {
        opts = opts || {}
        this.batchSize = opts.batchSize || 64 //batch size
        this.imgSize = opts.imgSize || 64
        this.imgC = opts.imgC || 3
        this.dim = 100 //input dimensions
        this.ngf = 64 // # generator filters 1st conv
        this.ndf = 64 // # dicriminator filters 1st conv
        this.lr = 0.0002 // learning rate (adam)
        this.beta1 = 0.5 // momentum (adam)
        this.D = null //Discriminator
        this.G = null //Generator
    }

    noise = (n = 1) => {
        return tf.randomUniform([n, this.dim], -1, 1)
    }

    generator = () => {
        if(this.G) {return this.G}

        let mult = this.imgSize / 8

        this.G = tf.sequential()

        this.G.add(tf.layers.dense({
            units: 4 * 4 * this.ngf * mult,
            kernelInitializer: 'glorotUniform',
            inputDim: this.dim
        }))
        this.G.add(tf.layers.reshape({targetShape: [4, 4, this.ngf * mult]}))
        this.G.add(tf.layers.batchNormalization({momentum: 0.5}))
        this.G.add(tf.layers.elu())

        while(mult > 1){
            this.G.add(tf.layers.upSampling2d({}))
            this.G.add(tf.layers.conv2dTranspose({
                filters: this.ngf * mult, 
                kernelSize: 4, 
                padding: 'same'
            }))
            this.G.add(tf.layers.batchNormalization({momentum: 0.5}))
            this.G.add(tf.layers.elu())

            mult /= 2
        }

        this.G.add(tf.layers.upSampling2d({}))
        this.G.add(tf.layers.conv2dTranspose({filters: this.imgC, kernelSize: 4, padding: 'same'}))
        this.G.add(tf.layers.activation({activation: 'tanh'}))

        const optimizer = tf.train.adam({learningRate: this.lr, beta1: this.beta1})
        this.G.compile({
            optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        })

        this.G.summary()

        return this.G
    }

    discriminator = () => {
        if(this.D) {return this.D}

        const inputShape = [this.imgSize, this.imgSize, this.imgC]
    
        this.D = tf.sequential()

        this.D.add(tf.layers.conv2d({
            filters: this.ndf,
            kernelInitializer: 'glorotUniform',
            kernelSize: 4, 
            strides: 2, 
            inputShape, 
            padding: 'same'
        }))
        this.D.add(tf.layers.leakyReLU({alpha: 0.2}))

        let [mult, newImgSize] = [1, this.imgSize / 2]

        while(newImgSize > 4){
            this.D.add(tf.layers.conv2d({
                filters: this.ndf * mult,
                kernelSize: 4, 
                strides: 2, 
                padding: 'same'
            }))
            this.D.add(tf.layers.batchNormalization())
            this.D.add(tf.layers.leakyReLU({alpha: 0.2}))

            mult *= 2
            newImgSize /= 2
        }

        this.D.add(tf.layers.conv2d({
            filters: this.ndf * mult, 
            kernelSize: 4, 
            strides: 1
        }))
        this.D.add(tf.layers.activation({activation: 'sigmoid'}))

        const optimizer = tf.train.adam({learningRate: this.lr, beta1: this.beta1})
        this.D.compile({
            optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        })

        this.D.summary()

        return this.D
        
    }
}

export default DCGAN