package demo

import (
	"fmt"
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
