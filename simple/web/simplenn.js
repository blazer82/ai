// Define activation function
function sigmoid(x, deriv) {
	if (deriv) {
		return x.multiply(x.multiply(-1).add(1))
	}
	return x.multiply(-1).exp().add(1).pow(-1)
}

// Initialize X, Y and weights
X = nj.zeros([4,3])
Y = nj.zeros([4,1])
y = nj.zeros([4,1])
w0 = nj.zeros([3,1])

// Initialize with real values
function init() {
	X = nj.array([
		[0,0,1],
		[0,1,1],
		[1,0,1],
		[1,1,1],
	])

	Y = nj.array([[0,0,1,1]]).transpose()

	y = nj.zeros([4,1])

	w0 = nj.random([3,1]).subtract(1).multiply(2)
	render()
}

function step(steps) {
	var steps = steps || 1

	for (var i = 0; i < steps; i++) {
		l0 = X
		l1 = y = sigmoid(l0.dot(w0))

		l1_error = Y.subtract(l1)

		l1_delta = l1_error.multiply(sigmoid(l1, true))

		w0 = w0.add(l0.transpose().dot(l1_delta))
	}

	console.log(l1)
	render()
}

function predict() {
	l0 = X
	y = sigmoid(l0.dot(w0))
	render()
}

function render() {
	function renderMatrix(M, tableName) {
		$('#'+tableName+' td').each(function(i, td) {
			row = Math.floor(i / M.shape[1])
			col = Math.floor(i % M.shape[1])
			$(td).html(M.get(row, col))
		})
	}
	renderMatrix(X, 'X')
	renderMatrix(Y, 'Y')
	renderMatrix(w0, 'w0')
	renderMatrix(y, 'y')
}

$(function() {
	init()
})
