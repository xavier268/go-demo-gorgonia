package demo

import (
	"fmt"
	"io/ioutil"
	"log"
	"testing"

	. "gorgonia.org/gorgonia"
)

// Same graph is used for all tests.
func makeGraph() (g *ExprGraph, z *Node) {
	g = NewGraph()

	var x, y *Node
	var err error

	// Define expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z, err = Mul(x, y)

	if err != nil {
		log.Fatal(err)
	}

	// Set initial values and run
	Let(x, 2.)
	Let(y, 2.5)

	return g, z
}

// TestBasic computes no gradient.
func TestBasic(t *testing.T) {
	var err error

	g, z := makeGraph()

	// Create a VM to run the program on
	machine := NewTapeMachine(g)
	defer machine.Close()

	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nThe z value is : %v\n", z.Value())

	// Writing graph as dot file
	ioutil.WriteFile("basic.dot", []byte(g.ToDot()), 0644)
}

// TestAutodiff showcases automatic differentiation,
// computes all possible gradients.
// Lisp machine is required.
func TestAutodiff(t *testing.T) {

	var err error

	g, z := makeGraph()
	x, y := g.ByName("x")[0], g.ByName("y")[0]

	// by default, LispMachine performs forward mode and backwards mode execution
	m := NewLispMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("z: %v\n", z.Value())

	if xgrad, err := x.Grad(); err == nil {
		fmt.Printf("dz/dx: %v\n", xgrad)
	}

	if ygrad, err := y.Grad(); err == nil {
		fmt.Printf("dz/dy: %v\n", ygrad)
	}

	// Writing graph as dot file
	ioutil.WriteFile("basic_autodiff.dot", []byte(g.ToDot()), 0644)
}

// testSymbolicDiff computes the requested gradients.
// tapeMachine is enough.
func TestSymbolicDiff(t *testing.T) {
	var err error

	g, z := makeGraph()
	x, y := g.ByName("x")[0], g.ByName("y")[0]

	// symbolically differentiate z with regards to x and y
	// this adds the gradient nodes to the graph g
	// grads is an array of the gradient wrt z
	// Gradient call also be read directly from x ...
	var grads Nodes
	grads, err = Grad(z, x, y)
	if err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)

	// then run
	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	fmt.Printf("z: %v\n", z.Value())

	xgrad, err := x.Grad() // xgrad access the gradient, IFF it was previously computed from grad()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("dz/dx: %v\n", xgrad)

	// Check that direct access to grads gives the same value as above ...
	if xg := grads[0].Value(); xg != xgrad {
		fmt.Printf("dz/dx has different values %v and %v\n", xgrad, xg)
		t.FailNow()
	}

	ygrad, err := y.Grad()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("dz/dy: %v\n", ygrad)

	if yg := grads[1].Value(); yg != ygrad {
		fmt.Printf("dz/dy has different values %v and %v\n", ygrad, yg)
		t.FailNow()
	}

	// Writing graph as dot file
	ioutil.WriteFile("basic_symbolicdiff.dot", []byte(g.ToDot()), 0644)

}
