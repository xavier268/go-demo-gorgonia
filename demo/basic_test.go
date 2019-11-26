package demo

import (
	"fmt"
	"io/ioutil"
	"log"
	"testing"

	. "gorgonia.org/gorgonia"
)

func TestBasic(t *testing.T) {

	g := NewGraph()

	var x, y, z *Node
	var err error

	// Define expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// Crete a VM to run the program on
	machine := NewTapeMachine(g)
	defer machine.Close()

	// Set initial values and run
	Let(x, 2.)
	Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nThe z value is : %v\n", z.Value())

}

// TestAutodiff showcases automatic differentiation
func TestAutodiff(t *testing.T) {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)

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
}

func TestSymbolicDiff(t *testing.T) {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z, err = Add(x, y)
	if err != nil {
		log.Fatal(err)
	}

	// symbolically differentiate z with regards to x and y
	// this adds the gradient nodes to the graph g
	// grads is an array of the gradient wrt z
	var grads Nodes
	grads, err = Grad(z, x, y)
	if err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	fmt.Printf("z: %v\n", z.Value())

	xgrad, err := x.Grad() // xgrad access the gradient, IFF it was previously computed from grad()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("dz/dx: %v\n", xgrad)

	ygrad, err := y.Grad()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("dz/dy: %v\n", ygrad)

	// Display all the gradients taht were calculated ...
	fmt.Printf("\nGrads : %v\n", grads)

	// Writing graph as dot file
	ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
}
