import * as tf from '@tensorflow/tfjs'

// tf.setBackend('cpu')

class DCGAN {
    constructor(opts) {
        opts = opts || {}
        this.imgSize = opts.imgSize
        this.imgC = opts.imgC

        this.dim = 100 //input dimensions
        this.ngf = 64 // # generator filters 1st conv
        this.ndf = 32 // # dicriminator filters 1st conv
        this.glr = 0.002 // learning rate (adam)
        this.dlr = 0.001 // learning rate (adam)
        this.beta1 = 0.5 // momentum (adam)
        this.D = null //Discriminator
        this.G = null //Generator
        this.DM = null
        this.AM = null
    }

    noise = (n = 1) => {
        return tf.randomUniform([n, this.dim], -1, 1)
    }

    getDim = () => {
        const d = this.imgSize / 4
        return d % 4 ? d : 4
    }

    generator = () => {
        if(this.G) {return this.G}

        let dim = this.getDim()
        let mult = Math.floor(this.imgSize / 8)

        this.G = tf.sequential()

        this.G.add(tf.layers.dense({
            units: dim * dim * (this.ngf * mult),
            kernelInitializer: 'glorotUniform',
            inputDim: this.dim,
            activation: 'tanh'
        }))
        this.G.add(tf.layers.reshape({targetShape: [dim, dim, this.ngf * mult]}))
        this.G.add(tf.layers.conv2dTranspose({
            filters: this.ngf * mult,
            kernelSize: 4,
            padding: 'same'
        }))
        this.G.add(tf.layers.batchNormalization())
        // this.G.add(tf.layers.elu())
        this.G.add(tf.layers.leakyReLU({alpha: 0.2}))

        while(mult >= 1){
            // this.G.add(tf.layers.upSampling2d({}))
            this.G.add(tf.layers.conv2dTranspose({
                filters: this.ngf * mult / 2, 
                kernelSize: 4, 
                strides: 2,
                padding: 'same'
            }))
            this.G.add(tf.layers.batchNormalization())
            // this.G.add(tf.layers.elu())
            this.G.add(tf.layers.leakyReLU({alpha: 0.2}))

            mult = Math.floor(mult / 2)
        }

        // this.G.add(tf.layers.upSampling2d({}))
        this.G.add(tf.layers.conv2dTranspose({filters: this.imgC, kernelSize: 4, padding: 'same'}))
        this.G.add(tf.layers.activation({activation: 'sigmoid'}))

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
        // this.D.add(tf.layers.elu())
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
            // this.D.add(tf.layers.elu())
            this.D.add(tf.layers.leakyReLU({alpha: 0.2}))

            mult *= 2
            newImgSize /= 2
        }

        this.D.add(tf.layers.conv2d({
            filters: this.ndf * mult, 
            kernelSize: 4, 
            strides: 1
        }))
        // this.D.add(tf.layers.elu())
        this.D.add(tf.layers.leakyReLU({alpha: 0.2}))

        this.D.add(tf.layers.flatten())
        this.D.add(tf.layers.dense({units: 1}))
        this.D.add(tf.layers.activation({activation: 'sigmoid'}))

        // this.D.summary()

        return this.D
        
    }

    dicriminatorModel = () => {
        if(this.DM) {return this.DM}
        // const optimizer = tf.train.rmsprop({learningRate: this.dlr})
        // const optimizer = tf.train.adam({learningRate: this.dlr, beta1: this.beta1})
        const optimizer = tf.train.adam()
        this.DM = tf.sequential()
        this.DM.add(this.discriminator())
        this.DM.compile({
            optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        })

        return this.DM
    }

    adversarialModel = () => {
        if(this.AM) {return this.AM}
        // const optimizer = tf.train.rmsprop({learningRate: this.glr})
        // const optimizer = tf.train.adam({learningRate: this.glr, beta1: this.beta1})
        const optimizer = tf.train.adam()
        this.discriminator().trainable = false
        this.AM = tf.sequential()
        this.AM.add(this.generator())
        this.AM.add(this.discriminator())
        this.AM.compile({
            optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        })

        // this.AM.summary()

        return this.AM
    }
}

export default DCGAN