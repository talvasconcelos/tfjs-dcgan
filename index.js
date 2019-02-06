import './style'
import { Component } from 'preact'
import * as tf from '@tensorflow/tfjs'

import {/*IMAGE_H, IMAGE_W,*/ MnistData} from './mnist'
import mnistDCGAN from './mnistgan'

import Samples from './components/samples'

export default class App extends Component {

	state = {
		data: null,
		dcgan: null
	}

	train = async (epochs = 100) => {
		const g = this.state.dcgan
		await this.setState({
			samples: g.gen.predict(this.state.noise)
		})
		await tf.nextFrame()
		for (let i = 0; i < epochs; i++) {
			let noise =	g.gan.noise(32)
			let fakes = g.gen.predict(noise)
			let real = this.state.data.nextTrainBatch(32).xs.reshapeAs(fakes)
			let x = tf.concat([real, fakes])
			let y = tf.concat([g.ONES_CAP, g.ZEROS])
			
			let dLoss = await g.discriminator.trainOnBatch(x, y)
			await tf.nextFrame()
			console.log('discriminator done')
			y = g.ONES
			noise = g.gan.noise(32)

			// console.log(y.shape)
			// console.log(noise.shape)

			let aLoss = await g.adversarial.trainOnBatch(noise, y)
			console.log('adversarial done')
			// console.log(aLoss)
			await tf.nextFrame()
			if(i % 10 === 0){
				await this.setState({
					samples: g.gen.predict(this.state.noise)
				})
			}
			
			console.log('samples update')
			
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
			}).then(() => {
				const noise = this.state.dcgan.gan.noise(9)
				this.setState({
					noise
				})
			})
	}

	render({}, {samples}) {
		return (
			<div>
				<h1>Hello, World!</h1>
				<button onClick={this.startTrain}>Train</button>
				{samples && <Samples examples={samples} />}
			</div>
		)
	}
}
