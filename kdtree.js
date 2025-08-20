export class KDTree {
  constructor (axesExtractors, points, asserts = false) {
    this.axes = axesExtractors
    this.asserts = asserts
    this.tree = this.buildTree(points, 0, asserts)
  }

  distance (a, b) {
    let sum = 0
    for (const axis of this.axes) {
      sum += (axis(a) - axis(b)) ** 2
    }
    return Math.sqrt(sum)
  }

  buildTree (points, axisIndex, asserts = false) {
    if (points.length === 0)
      return null
    const axisExtractor = this.axes[axisIndex]
    const comparator = (a, b) => axisExtractor(a) - axisExtractor(b)
    const sorted = [...points]
    sorted.sort(comparator)
    let pivotIndex = Math.floor(sorted.length / 2)
    const point = sorted[pivotIndex]
    const nextAxisIndex = (axisIndex + 1) % this.axes.length
    let leftPoints = sorted.slice(0, pivotIndex)
    let rightPoints = sorted.slice(pivotIndex + 1, sorted.length)
    if (asserts) {
      console.assert(leftPoints.length + rightPoints.length + 1 === points.length)
    }
    let result = {
      point,
      left: this.buildTree(leftPoints, nextAxisIndex, asserts),
      right: this.buildTree(rightPoints, nextAxisIndex, asserts),
      axis: axisExtractor
    }
    if (asserts) {
      const leftContent = this.getContents(result.left)
      console.assert(leftContent.every(p => comparator(p, result.point) <= 0))
      const rightContent = this.getContents(result.right)
      console.assert(rightContent.every(p => comparator(p, result.point) >= 0))
      console.assert(leftContent.length + rightContent.length + 1 === points.length)
    }
    return result
  }

  getContents (node, collector = null) {
    if (collector == null)
      collector = []
    if (node == null)
      return collector
    this.getContents(node.left, collector)
    collector.push(node.point)
    this.getContents(node.right, collector)
    return collector
  }

  findNN (point) {
    return this._findNNInTree(point, this.tree)
  }

  _findNNInTree (needdlePoint, node = null) {
    if (node == null)
      return null
    const closestChoice = (p1, p2) => {
      if (p1 == null)
        return p2
      if (p2 == null)
        return p1
      return this.distance(needdlePoint, p1) < this.distance(needdlePoint, p2) ? p1 : p2
    }
    const [pointValue, refValue] = [node.axis(needdlePoint), node.axis(node.point)]
    const children = [node.left, node.right]
    if (pointValue >= refValue)
      children.reverse()
    let best = closestChoice(this._findNNInTree(needdlePoint, children[0]), node.point)
    // need to check the other side of the wall for a hidden creeper if said wall is closer than the current nearest neighbor
    if (this.distance(needdlePoint, best) >= Math.abs(node.axis(needdlePoint) - node.axis(node.point))) {
      best = closestChoice(this._findNNInTree(needdlePoint, children[1]), best)
    }
    return best
  }
}
