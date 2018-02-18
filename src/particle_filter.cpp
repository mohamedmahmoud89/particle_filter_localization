/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;		

	// This line creates a normal (Gaussian) distribution
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	num_particles = 8;
	for (int i = 0; i < num_particles; ++i) 
	{
		struct Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double tmp_x,tmp_y,tmp_theta;

	for (int i = 0; i < num_particles; ++i) 
	{
		if(abs(yaw_rate) >= 0.0001)
		{
			tmp_x = particles[i].x + ((velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t))-sin(particles[i].theta)));
			tmp_y = particles[i].y + ((velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta+(yaw_rate*delta_t))));
			tmp_theta = particles[i].theta +(yaw_rate*delta_t);
		}
		else
		{
			tmp_x = particles[i].x + (velocity*delta_t)*cos(particles[i].theta);
			tmp_y = particles[i].y + (velocity*delta_t)*sin(particles[i].theta);
			tmp_theta = particles[i].theta;
		}
		
		normal_distribution<double> dist_x(tmp_x, std_pos[0]);
		normal_distribution<double> dist_y(tmp_y, std_pos[1]);
		normal_distribution<double> dist_theta(tmp_theta, std_pos[2]);
		
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double dist_1,dist_2,dist,min_dist;
	for(int o_idx=0; o_idx < observations.size(); o_idx++)
	{
		min_dist = 65000;
		for(int p_idx=0; p_idx < predicted.size(); p_idx++)
		{
			dist_1 = predicted[p_idx].x - observations[o_idx].x;
			dist_1 *= dist_1;
			dist_2 = predicted[p_idx].y - observations[o_idx].y;
			dist_2 *= dist_2;
			dist = sqrt(dist_1+dist_2);
			if (dist < min_dist)
			{
				min_dist = dist;
				observations[o_idx].id = predicted[p_idx].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	std::vector<LandmarkObs> prj_observations;
	std::vector<LandmarkObs> map_lm;
	LandmarkObs lm;
	const long double pi = 3.141592653589793238L;
	
	weights.clear();
	for(int p_idx=0; p_idx < num_particles; p_idx++)
	{ 
		prj_observations.clear();
		map_lm.clear();
		for(int o_idx=0; o_idx < observations.size(); o_idx++)
		{	
			lm.id = 0;
			lm.x = particles[p_idx].x + (observations[o_idx].x*cos(particles[p_idx].theta)) - (observations[o_idx].y*sin(particles[p_idx].theta));
			lm.y = particles[p_idx].y + (observations[o_idx].x*sin(particles[p_idx].theta)) + (observations[o_idx].y*cos(particles[p_idx].theta));
			prj_observations.push_back(lm);
		}
		
		for(int o_idx=0; o_idx < map_landmarks.landmark_list.size(); o_idx++)
		{
			lm.id = map_landmarks.landmark_list[o_idx].id_i;
			lm.x = map_landmarks.landmark_list[o_idx].x_f;
			lm.y = map_landmarks.landmark_list[o_idx].y_f;
			map_lm.push_back(lm);
		}
		
		dataAssociation(map_lm,prj_observations);
		
		particles[p_idx].weight = 1.0;
		
		for(int o_idx=0; o_idx < prj_observations.size(); o_idx++)
		{
			for(int m_idx=0; m_idx < map_lm.size(); m_idx++)
			{
				if(prj_observations[o_idx].id == map_landmarks.landmark_list[m_idx].id_i)
				{
					particles[p_idx].weight *= (1/(2.0*pi*std_landmark[0]*std_landmark[1]));
					double expo = 0;
					expo -= 0.5*(pow(((prj_observations[o_idx].x - map_lm[m_idx].x)/std_landmark[0]),2));
					expo -= 0.5*(pow(((prj_observations[o_idx].y - map_lm[m_idx].y)/std_landmark[1]),2));
					particles[p_idx].weight *= exp(expo);
					break;
				}
			}
		}
		
		weights.push_back(particles[p_idx].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	std::uniform_int_distribution<unsigned int> index_dist(0, num_particles-1);
	unsigned int index = index_dist(gen);
	double B=0;
	double max_weight = *max_element(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	std::uniform_real_distribution<double> weight_dist(0, 2*max_weight);
	
	for(int idx=0; idx < num_particles; idx++)
	{
		B += weight_dist(gen);
		
		while(particles[index].weight < B)
		{
			B -= particles[index].weight;
			index = (index+1)%num_particles;
		}
		
		new_particles.push_back(particles[index]);
	}
	
	particles.clear();
	
	for(int idx=0; idx < new_particles.size(); idx++)
	{
		particles.push_back(new_particles[idx]);			
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
