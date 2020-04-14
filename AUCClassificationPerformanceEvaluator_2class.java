/*
 *    WindowClassificationPerformanceEvaluator.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package src;

import java.util.TreeSet;

import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;

import moa.evaluation.*;
import moa.tasks.TaskMonitor;
import weka.core.Utils;
import weka.core.Instance;


/**
 * Classification evaluator that updates evaluation results using a sliding
 * window. Only to be used for binary classification problems with unweighted instances.
 * 
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class AUCClassificationPerformanceEvaluator_2class extends AbstractOptionHandler {

  private static final long serialVersionUID = 1L;

  public int widthOption = 500;//Size of Window

  protected double totalObservedInstances = 0;

  protected Estimator aucEstimator;

  protected int numClasses;

  public class Estimator {

    public class Score implements Comparable<Score> {
      /**
       * Predicted score of the example
       */
      protected double value;

      /**
       * Age of example - position in the window where the example was
       * added
       */
      protected int posWindow;

      /**
       * True if example's true label is positive
       */
      protected boolean isPositive;

      /**
       * Constructor.
       * 
       * @param value
       *            score value
       * @param position
       *            score position in window (defines its age)
       * @param isPositive
       *            true if the example's true label is positive
       */
      public Score(double value, int position, boolean isPositive) {
    	  this.value = value;
    	  this.posWindow = position;
    	  this.isPositive = isPositive;
      }

      /**
       * Sort descending based on score value.
       */
      @Override
      public int compareTo(Score o) {
    	  if (o.value < this.value) {
    		  return -1;
    	  } else if (o.value > this.value){
    		  return 1;
    	  } else {
    		  if (o.posWindow > this.posWindow) {
    			  return -1;
    		  } else if (o.posWindow < this.posWindow){
    			  return 1;
    		  } else {
    			  return 0;
    		  }
    	  }
      }

      @Override
      public boolean equals(Object o) {
    	  return (o instanceof Score) && ((Score)o).posWindow == this.posWindow;
      }
    }

    protected TreeSet<Score> sortedScores;//the tree for calculating prequential AUC

    protected TreeSet<Score> holdoutSortedScores;//for calculating HoldoutAUC

    protected Score[] window;

    protected double[] predictions;//if the example is correctly classified or not: 1-correct, 0-incorrect. 

    protected int posWindow;//the position where the oldest example that needs to be removed (if window is full) and where the new example that needs to be added.

    protected int size;//window size

    protected double numPos;

    protected double numNeg;

    protected double holdoutNumPos;

    protected double holdoutNumNeg;

    protected double correctPredictions;

    protected double[] columnKappa;

    protected double[] rowKappa;

    public Estimator(int sizeWindow) {
      this.sortedScores = new TreeSet<Score>();
      this.holdoutSortedScores = new TreeSet<Score>();
      this.size = sizeWindow;
      this.window = new Score[sizeWindow];
      this.predictions = new double[sizeWindow];

      this.rowKappa = new double[numClasses];
      this.columnKappa = new double[numClasses];
      for (int i = 0; i < numClasses; i++) {
	this.rowKappa[i] = 0.0;
	this.columnKappa[i] = 0.0;
      }

      this.posWindow = 0; 
      this.numPos = 0;
      this.numNeg = 0;
      this.holdoutNumPos = 0;
      this.holdoutNumNeg = 0;
      this.correctPredictions = 0;
    }

    public void add(double score, boolean isPositive, boolean correctPrediction) {
    	// // periodically update holdout evaluation, for "getHoldoutAUC"
    	if (size > 0 && posWindow % this.size == 0) {
    		this.holdoutSortedScores = new TreeSet<Score>();

    		for (Score s : this.sortedScores) {
    			this.holdoutSortedScores.add(s);
    		}

    		this.holdoutNumPos = this.numPos;
    		this.holdoutNumNeg = this.numNeg;
    	}

    	// // if the window is used and it's full			
    	if (size > 0 && posWindow >= this.size) {
    		// // remove the oldest example
    		sortedScores.remove(window[posWindow % size]);
    		correctPredictions -= predictions[posWindow % size];

    		if (window[posWindow % size].isPositive) {
    			numPos--;
    		} else {
    			numNeg--;
    		}

    		int oldestExampleTrueClass = window[posWindow % size].isPositive ? 0 : 1;
    		int oldestExamplePredictedClass = predictions[posWindow % size] == 1.0 ? oldestExampleTrueClass : Math.abs(oldestExampleTrueClass - 1);  

    		this.rowKappa[oldestExamplePredictedClass] -= 1;
    		this.columnKappa[oldestExampleTrueClass] -= 1;
    	}

    	// // add new example
    	Score newScore = new Score(score, posWindow, isPositive);
    	sortedScores.add(newScore);
    	correctPredictions += correctPrediction ? 1 : 0;

    	int trueClass = isPositive ? 0 : 1;
    	int predictedClass = correctPrediction ? trueClass : Math.abs(trueClass - 1);  
    	this.rowKappa[predictedClass] += 1;
    	this.columnKappa[trueClass] += 1;

    	if (newScore.isPositive) {
    		numPos++;
    	} else {
    		numNeg++;
    	}

    	if (size > 0) {
    		window[posWindow % size] = newScore;
    		predictions[posWindow % size] = correctPrediction ? 1 : 0;
    	}

    	//// posWindow needs to be always incremented to differentiate between examples in the red-black tree
    	posWindow++;
    }

    public double getAUC() {
    	double AUC = 0;
    	double c = 0;

    	if (numPos == 0 || numNeg == 0) {
    		return 1;
    	}

      for (Score s : sortedScores){
	if(s.isPositive) {
	  c += 1;
	} else {
	  AUC += c;
	}
      }

      return AUC / (numPos * numNeg);
    }

    public double getHoldoutAUC() {
      double AUC = 0;
      double c = 0;

      if (holdoutSortedScores.isEmpty()) {
	return 0;
      }

      if (holdoutNumPos == 0 || holdoutNumNeg == 0) {
	return 1;
      }

      for (Score s : holdoutSortedScores){
	if(s.isPositive) {
	  c += 1;
	} else {
	  AUC += c;
	}
      }

      return AUC / (holdoutNumPos * holdoutNumNeg);
    }

    public double getScoredAUC() {
      double AOC = 0;
      double AUC = 0;
      double r = 0;
      double c = 0;
      double R_plus, R_minus;

      if (numPos == 0 || numNeg == 0) {
	return 1;
      }

      for (Score s : sortedScores){
	if(s.isPositive) {
	  c += s.value;
	  AOC += r;
	} else {
	  r += s.value;
	  AUC += c;
	}
      }

      R_minus = (numPos*r - AOC)/(numPos * numNeg);
      R_plus = (AUC)/(numPos * numNeg);		
      return R_plus - R_minus;
    }

    public double getRatio() {
      if(numNeg == 0) {
	return Double.MAX_VALUE;
      } else {
	return numPos/numNeg;
      }
    }

    public double getAccuracy() {
      if (size > 0) {
	return totalObservedInstances > 0.0 ? correctPredictions / Math.min(size, totalObservedInstances) : 0.0;
      } else {
	return totalObservedInstances > 0.0 ? correctPredictions / totalObservedInstances : 0.0;
      }
    }

    public double getKappa() {
      double p0 = getAccuracy();
      double pc = 0.0;

      if (size > 0) {
	for (int i = 0; i < numClasses; i++) {
	  pc += (this.rowKappa[i]/Math.min(size, totalObservedInstances)) * (this.columnKappa[i]/Math.min(size, totalObservedInstances));
	}
      } else {
	for (int i = 0; i < numClasses; i++) {
	  pc += (this.rowKappa[i]/totalObservedInstances) * (this.columnKappa[i]/totalObservedInstances);
	}
      }
      return (p0 - pc) / (1.0 - pc);
    }
  }


  public void reset(int numClasses) {
    if (numClasses != 2) {
      throw new RuntimeException(
	  "Too many classes ("
	      + numClasses
	      + "). AUC evaluation can be performed only for two-class problems!");
    }

    this.numClasses = numClasses;

    this.aucEstimator = new Estimator(this.widthOption);
    this.totalObservedInstances = 0;
  }

  
  public void addResult(Instance inst, double[] classVotes) {

	  double weight = inst.weight();
	  int trueClass = (int) inst.classValue();

	  if (weight > 0.0) {
		  // // initialize evaluator
		  if (totalObservedInstances == 0) {
			  reset(inst.dataset().numClasses());
		  }
		  this.totalObservedInstances += 1;

		  Double normalizedVote = 1.0;

		  //// normalize and add score
		  if(classVotes.length == 2) {
			  normalizedVote = classVotes[0]/(classVotes[0] + classVotes[1]);
		  }

		  if(normalizedVote.isNaN()){
			  normalizedVote = 0.0;
		  }

		  this.aucEstimator.add(normalizedVote, trueClass == 0, Utils.maxIndex(classVotes) == trueClass);//for 2-class case, treat class0 as the positive class by default
	  }
  }


  public Measurement[] getPerformanceMeasurements() {
    return new Measurement[] {
	new Measurement("classified instances",
	    this.totalObservedInstances),			
	    new Measurement("AUC", this.aucEstimator.getAUC()),
	    new Measurement("sAUC", this.aucEstimator.getScoredAUC()),
	    new Measurement("Accuracy", this.aucEstimator.getAccuracy()),
	    new Measurement("Kappa", this.aucEstimator.getKappa()),
	    new Measurement("Periodical holdout AUC", this.aucEstimator.getHoldoutAUC()),
	    new Measurement("Pos/Neg ratio", this.aucEstimator.getRatio())};

  }

  @Override
  public void getDescription(StringBuilder sb, int indent) {
    Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
	sb, indent);
  }

  @Override
  public void prepareForUseImpl(TaskMonitor monitor,
      ObjectRepository repository) {
  }

  public Estimator getAucEstimator() {
    return aucEstimator;
  }


}
