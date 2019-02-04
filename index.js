import './style'
import { Component } from 'preact'
import * as tf from '@tensorflow/tfjs'

import {/*IMAGE_H, IMAGE_W,*/ MnistData} from './mnist'
import mnistDCGAN from './mnistgan'

export default class App extends Component {

	state = {
		data: null,
		dcgan: null
	}

	train = async (epochs = 50) => {
		const g = this.state.dcgan
		for (let i = 0; i < epochs; i++) {
			let noise =	g.gan.noise(32)
			let fakes = g.gen.predict(noise)
			let real = this.state.data.nextTrainBatch(32).xs.reshapeAs(fakes)
			let x = tf.concat([real, fakes])
			let y = tf.concat([g.ONES_CAP, g.ZEROS])

			let dLoss = await g.discriminator.trainOnBatch(x, y)
			await tf.nextFrame()

			y = g.ONES
			noise = g.gan.noise(32)

			let aLoss = await g.adversarial.trainOnBatch(noise, y)
			await tf.nextFrame()
			
			console.log(dLoss, aLoss)
		}
	}

	startTrain = () => {
		console.log('Start...')
		this.train()
	}

	componentDidMount = () => {
		const data = new MnistData()
		data.load()
		.then(() => {
			this.setState({
				data,
				dcgan: new mnistDCGAN({
					imgSize: 28,
					imgC: 1,
					batchSize: 32,
					data
				})
			})
			})/*.then(() => {
				// this.state.dcgan.trainBatch()
				// const gen = this.state.dcgan.generator()
				// const noise = this.state.dcgan.noise(1)
				// noise.print()
				// const p = gen.predict(noise)
				// p.print()
			})*/
	}

	render() {
		return (
			<div>
				<h1>Hello, World!</h1>
				<button onClick={this.startTrain}>Train</button>
			</div>
		)
	}
}
