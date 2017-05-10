/* propagate.cl */
__kernel void propagate(
	__global float4* position,
	__global float4* newPosition,
	__global float4* velocity,
	__global float4* newVelocity,
	__private const float tau,
	__private const float A,
	__private const float a,
	__private const float B,
	__private const float b,
	__private const float C,
	__private const float c,
	__private const float timeStep,
	const int count
)
{
	// pick a particle for which we will calculate the new position
	float4 pos = position[count];
	float4 vel = velocity[count];

	// initilize local flow velocity to 0
	float4 flow_vel = (float4)(0.0, 0.0, 0.0, 0.0);

	// calculate local flow velocity
	flow_vel.x = A * cos(a * (pos.x + (3.14159 / (a * 2)))) * sin(b * (pos.y + (3.14159 / (a * 2)))) * sin(c * (pos.z + (3.14159 / (a * 2))));
	flow_vel.y = B * sin(a * (pos.x + (3.14159 / (a * 2)))) * cos(b * (pos.y + (3.14159 / (a * 2)))) * sin(c * (pos.z + (3.14159 / (a * 2))));
	flow_vel.z = C * sin(a * (pos.x + (3.14159 / (a * 2)))) * sin(b * (pos.y + (3.14159 / (a * 2)))) * cos(c * (pos.z + (3.14159 / (a * 2))));
	
	// velocity loop
	vel.x += (timeStep / tau) * (flow_vel.x - vel.x); 
	vel.y += (timeStep / tau) * (flow_vel.y - vel.y) - 0.001*tau;
	vel.z += (timeStep / tau) * (flow_vel.z - vel.z);
	
	// displacement loop
	pos.x += (timeStep * (vel.x + velocity[count].x) / 2);
	pos.y += (timeStep * (vel.y + velocity[count].y) / 2);
	pos.z += (timeStep * (vel.z + velocity[count].z) / 2);
	
	// update values
	newPosition[count] = pos;
	newVelocity[count] = vel;
}

