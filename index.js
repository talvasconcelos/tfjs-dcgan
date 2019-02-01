import './style'
import { Component } from 'preact'

import {IMAGE_H, IMAGE_W, MnistData} from './mnist'
import DCGAN from './dcgan'

export default class App extends Component {

	state = {
		train_data: null,
		test_data: null,
		dcgan: new DCGAN()
	}

	componentDidMount = () => {
		const data = new MnistData()
		data.load().then(() => {
			this.setState({
				train_data: data.getTrainData(),
				test_data: data.getTestData()})
			}).then(() => {
				const gen = this.state.dcgan.generator()
				const noise = this.state.dcgan.noise(3)
				noise.print()
				const p = gen.predict(noise)
				p.print()
			})
	}

	render() {
		return (
			<div>
				<h1>Hello, World!</h1>
				
			</div>
		)
	}
}
