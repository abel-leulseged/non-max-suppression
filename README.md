<<<<<<< HEAD
# Non-Max Suppression

Algorithm:\
	Efficient and vectorized imlpementation
	In each step of the iteration, we choose the box with the highest score, calculate the IOU and remove all overlapping boxes whose IOU is above an overlap threshold (default = 0.45)
	All steps of the calculation are vectorized to increase efficiency. Final time for randomized test case is 5.8s

Test Case:\
	Added test functions for edge and general cases. The former includes return the empty list when all scores are below a minimum_threhold and general case is to remove overlapping boxes for a given class.
=======
# non-max-suppression
No
>>>>>>> 5388f8ac1e403f85795dc872a85b203c85ed3228
